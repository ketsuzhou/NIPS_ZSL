from functools import partial
import pdb

import torch
import torch.nn as nn
import numpy as np

from models.flow.invertible_net import *


class inn_classifier(nn.Module):
    def __init__(self, inn_prior_sampler, continuous_attributes):
        super(inn_classifier, self).__init__()
        self.dim_attribute = len(continuous_attributes)
        self.num_inn = len(inn_prior_sampler)
        self.dim_attribute_per_inn = self.dim_attribute / self.num_inn

        self.num_classes = int(continuous_attributes.shape[0])
        self.num_dim = int(continuous_attributes.size(1))

        mu_populate_dims = int(np.prod(self.dims))
        init_latent_scale = 5.0
        init_scale = init_latent_scale / \
            np.sqrt(2 * mu_populate_dims // self.num_classes)
        for k in range(mu_populate_dims // self.n_classes):
            self.mu.data[0, :, self.n_classes * k: self.n_classes *
                         (k+1)] = init_scale * torch.eye(self.n_classes)

        self.sigma = nn.Parameter(torch.ones(self.num_classes))

        self.trainable_params = list(self.invertible_net.parameters())
        self.trainable_params = list(
            filter((lambda p: p.requires_grad), self.trainable_params))
        weight_init = 1.0
        for p in self.trainable_params:
            p.data *= weight_init

        self.train_inn = True

        # self.trainable_params += [self.mu, self.phi]
        self.trainable_params += [self.sigma]
        base_lr = float(self.args.lr)

        optimizer_params = [{
            'params': list(filter(lambda p: p.requires_grad, self.invertible_net.parameters()))
        }, ]

        # if optimizer == 'SGD':
        #     self.optimizer = torch.optim.SGD(optimizer_params, base_lr,
        #                                      momentum=float(
        #                                          self.args['training']['sgd_momentum']),
        #                                      weight_decay=float(self.args['training']['weight_decay']))
        # elif optimizer == 'ADAM':
        #     self.optimizer = torch.optim.Adam(optimizer_params, base_lr,
        #                                       betas=self.args.train['adam_betas'],
        #                                       weight_decay=float(self.args.train['weight_decay']))
        # elif optimizer == 'AGGMO':
        #     import aggmo
        #     self.optimizer = aggmo.AggMo(optimizer_params, base_lr,
        #                                  betas=eval(
        #                                      self.args['training']['aggmo_betas']),
        #                                  weight_decay=float(self.args['training']['weight_decay']))
        # else:
        #     raise ValueError(f'what is this optimizer, {optimizer}?')

    def cluster_distances(self, z, y):
        # batchsize x n_classes
        z_i_z_i = torch.sum(z**2, dim=1, keepdim=True)
        mu_j_mu_j = torch.sum(self.mu**2, dim=2)         # 1 x n_classes
        # batchsize x n_classes
        z_i_mu_j = torch.mm(z, self.mu.squeeze().t())

        return -2 * z_i_mu_j + z_i_z_i + mu_j_mu_j

    def mu_pairwise_dist(self):
        mu_i_mu_j = self.mu.squeeze().mm(self.mu.squeeze().t())
        mu_i_mu_i = torch.sum(
            self.mu.squeeze()**2, 1, keepdim=True).expand(self.n_classes, self.n_classes)
        dist = mu_i_mu_i + mu_i_mu_i.t() - 2 * mu_i_mu_j
        return torch.masked_select(dist, (1 - torch.eye(self.n_classes).cuda()).bool()).clamp(min=0.)

    def forward(self, f_z, y):
        log_sigma = torch.log_softmax(self.sigma, dim=0).view(1, -1)
        dist2cluster_z = self.cluster_distances(f_z)

        # there is no need to normalizing p_f_z, because normalizer is a constant
        # log_p_f_z = torch.logsumexp(- 0.5 * dist2cluster_z + log_sigma, dim=1) / self.num_classes
        log_p_f_z = torch.logsumexp(
            dist2cluster_z / (2 * self.sigma**2), dim=1)

        # detach log_wy to block gradient
        log_sigma = log_sigma.detach()
        cross_entropy = torch.sum(
            torch.log_softmax(dist2cluster_z / (2 * self.sigma**2), dim=1) * y, dim=1)

        # todo
        logits_tr = - 0.5 * dist2cluster_z
        acc_tr = torch.mean(
            (torch.max(y, dim=1)[1]
             == torch.max(logits_tr.detach(), dim=1)[1]).float())

        return log_p_f_z, cross_entropy

    def validate(self, x, y, eval_mode=True):
        is_train = self.invertible_net.training
        if eval_mode:
            self.invertible_net.eval()

        with torch.no_grad():
            losses = self.forward(x, y, loss_mean=False)
            l_x, class_nll, l_y, logits, acc = (losses['L_x_tr'].mean(),
                                                losses['L_cNLL_tr'].mean(),
                                                losses['L_y_tr'].mean(),
                                                losses['logits_tr'],
                                                losses['acc_tr'])
            mu_dist = torch.mean(torch.sqrt(self.mu_pairwise_dist()))

        if is_train:
            self.invertible_net.train()

        return {'L_x_val':      l_x,
                'L_cNLL_val':   class_nll,
                'logits_val':   logits,
                'L_y_val':      l_y,
                'acc_val':      acc,
                'delta_mu_val': mu_dist}

    def sample(self, y, temperature=1.):
        z = temperature * torch.randn(y.shape[0], self.ndim_tot).cuda()
        mu = torch.sum(y.round().view(-1, self.n_classes, 1) * self.mu, dim=1)
        return self.invertible_net(z, rev=True)

    def save(self, fname):
        torch.save({'invertible_net': self.invertible_net.state_dict(),
                    'mu':  self.mu,
                    'phi': self.sigma,
                    'opt': self.optimizer.state_dict()}, fname)

    def load(self, fname):
        data = torch.load(fname)
        data['invertible_net'] = {
            k: v for k, v in data['invertible_net'].items() if 'tmp_var' not in k}
        self.invertible_net.load_state_dict(data['invertible_net'])
        self.mu.data.copy_(data['mu'].data)
        self.sigma.data.copy_(data['phi'].data)
        try:
            pass
        except:
            print('loading the optimizer went wrong, skipping')


class inn_intervention(nn.Module):
    def __init__(self, inn):
        super(inn_intervention, self).__init__()
        self.inn = inn

    def forward(self, sample, sourse_attributes, target_attributes):
        intervention = torch.abs(target_attributes - sourse_attributes)
        intervented_sample = self.inn(sample, c=intervention)
