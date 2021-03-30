import numpy as np
import torch

from torch import nn
from torch.nn import functional as F, init

from manifold_flow.transforms.linear import Linear


class LULinear(Linear):
    """A linear transform where we parameterize the LU decomposition of the weights.
        W = PL(U+diag(s)), where P is a permutation matrix, L is a lower triangular matrix with ones on the diagonal, U is an upper triangular matrix with zeros on the diagonal, and s is a learnable vector, which determinates the determinant of the one-to-one invertible convelutional layer
    """

    def __init__(self, features, using_cache=False, identity_init=True, eps=1e-3):
        super().__init__(features, using_cache)
        self.eps = eps
        num_triangular_entries_without_diagonal = ((features - 1) * features) // 2
        self.diag_indices = np.diag_indices(features)

        self.lower_indices_without_diagonal = np.tril_indices(features, k=-1)
        self.lower_entries_without_diagonal = nn.Parameter(
            torch.zeros(num_triangular_entries_without_diagonal)
        )
        # a=torch.zeros(12,12)
        # a[self.upper_indices]=1
        self.upper_indices_without_diagonal = np.triu_indices(features, k=1)
        self.upper_entries_without_diagonal = nn.Parameter(
            torch.zeros(num_triangular_entries_without_diagonal)
        )
        self.unconstrained_upper_diag = nn.Parameter(torch.zeros(features))

        self._initialize(identity_init)

    def _initialize(self, identity_init):
        init.zeros_(self.bias)

        if identity_init:
            init.zeros_(self.lower_entries_without_diagonal)
            init.zeros_(self.upper_entries_without_diagonal)
            constant = np.log(np.exp(1 - self.eps) - 1)
            init.constant_(self.unconstrained_upper_diag, constant)
        else:
            stdv = 1.0 / np.sqrt(self.features)
            init.uniform_(self.lower_entries_without_diagonal, -stdv, stdv)
            init.uniform_(self.upper_entries_without_diagonal, -stdv, stdv)
            init.uniform_(self.unconstrained_upper_diag, -stdv, stdv)

    def _create_lower_upper(self):
        """   W = PL(U+diag(s)), where P is a permutation matrix, L is a lower triangular matrix with ones on the diagonal, U is an upper triangular matrix with zeros on the diagonal, and s is a learnable vector, which determinates the determinant of the one-to-one invertible convelutional layer
        """        
        lower = self.lower_entries_without_diagonal.new_zeros(self.features, self.features)
        lower[self.lower_indices_without_diagonal[0], self.lower_indices_without_diagonal[1]] = self.lower_entries_without_diagonal
        # The diagonal of L is taken to be all-ones without loss of generality.
        lower[self.diag_indices[0], self.diag_indices[1]] = 1.0

        upper = self.upper_entries_without_diagonal.new_zeros(self.features, self.features)
        upper[self.upper_indices_without_diagonal[0], self.upper_indices_without_diagonal[1]] = self.upper_entries_without_diagonal
        # adding  diag(s) to U
        upper[self.diag_indices[0], self.diag_indices[1]] = self.upper_diag

        return lower, upper

    @property
    def upper_diag(self):
        return F.softplus(self.unconstrained_upper_diag) + self.eps

    def forward_no_cache(self, permuted_inputs, full_jacobian=False):
        """  inputs \times W = permuted_inputs \times L(U+diag(s)), L is a lower triangular matrix with ones on the diagonal, U is an upper triangular matrix with zeros on the diagonal, and s is a learnable vector
        Args:
            permuted_inputs ([b * h * w, c]): [description]
            full_jacobian (bool, optional): [description]. Defaults to False.
        Cost:
            output = O(D^2N)
            logabsdet = O(D)
        where:
            D = num of features
            N = num of inputs
        Returns:
            [type]: [description]
        """        
        lower, upper = self._create_lower_upper()
        outputs = F.linear(permuted_inputs, upper)
        outputs = F.linear(outputs, lower, self.bias)

        if full_jacobian:
            # outputs.shape[0] = b * h * w
            jacobian = torch.mm(lower, upper).unsqueeze(0).expand(outputs.size(0), -1, -1)
            return outputs, jacobian
        else:
            # outputs.shape[0] = b * h * w
            logabsdet = self.logabsdet() * permuted_inputs.new_ones(outputs.shape[0])
            return outputs, logabsdet

    def inverse_no_cache(self, inputs, full_jacobian=False):
        """Cost:
            output = O(D^2N)
            logabsdet = O(D)
        where:
            D = num of features
            N = num of inputs
        """
        lower, upper = self._create_lower_upper()
        outputs = inputs - self.bias
        outputs, _ = torch.triangular_solve(outputs.t(), lower, upper=False, unitriangular=True)
        outputs, _ = torch.triangular_solve(outputs, upper, upper=True, unitriangular=False)
        outputs = outputs.t()

        if full_jacobian:
            jacobian = torch.mm(lower, upper).inverse().unsqueeze(0).expand(outputs.size(0), -1, -1)  # TODO: make this faster
            return outputs, jacobian
        else:
            logabsdet = -self.logabsdet()
            logabsdet = logabsdet * inputs.new_ones(outputs.shape[0])
            return outputs, logabsdet

    def weight(self):
        """Cost:
            weight = O(D^3)
        where:
            D = num of features
        """
        lower, upper = self._create_lower_upper()
        return lower @ upper

    def weight_inverse(self):
        """Cost:
            inverse = O(D^3)
        where:
            D = num of features
        """
        lower, upper = self._create_lower_upper()
        identity = torch.eye(self.features, self.features)
        lower_inverse, _ = torch.trtrs(identity, lower, upper=False, unitriangular=True)
        weight_inverse, _ = torch.trtrs(lower_inverse, upper, upper=True, unitriangular=False)
        return weight_inverse

    def logabsdet(self):
        """Cost:
            logabsdet = O(D)
        where:
            D = num of features
        """
        return torch.sum(torch.log(self.upper_diag))
