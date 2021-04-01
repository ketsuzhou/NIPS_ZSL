import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_operations import OPS, EncCombinerCell, DecCombinerCell, Conv2D, get_skip_connection, SE
from torch.distributions.bernoulli import Bernoulli
from neural_ar_operations import ARConv2d
from utils import get_stride_for_cell_type, get_input_size, groups_per_scale, get_arch_cells
from distributions import Normal, DiscMixLogistic
from inplaced_sync_batchnorm import SyncBatchNormSwish
from generative_classifier import generative_classifier
from models.flow.invertible_net import invertible_net
import utils


class Cell(nn.Module):
    def __init__(self, Cin, Cout, cell_type, arch, use_se):
        super(Cell, self).__init__()
        self.cell_type = cell_type

        stride = get_stride_for_cell_type(self.cell_type)
        self.skip = get_skip_connection(Cin,
                                        stride,
                                        affine=False,
                                        channel_mult=2)
        self.use_se = use_se
        self._num_nodes = len(arch)
        self._ops = nn.ModuleList()
        for i in range(self._num_nodes):
            stride = get_stride_for_cell_type(self.cell_type) if i == 0 else 1
            C = Cin if i == 0 else Cout
            primitive = arch[i]
            op = OPS[primitive](C, Cout, stride)
            self._ops.append(op)
        # SE
        if self.use_se:
            self.se = SE(Cout, Cout)

    def forward(self, s):
        # skip branch
        skip = self.skip(s)
        for i in range(self._num_nodes):
            s = self._ops[i](s)

        s = self.se(s) if self.use_se else s
        return skip + 0.1 * s


class Im_AutoEncoder(nn.Module):
    def __init__(self, args, writer, arch_instance):
        super(Im_AutoEncoder, self).__init__()
        # self.writer = writer
        self.arch_instance = arch_instance
        self.in_shape = args.in_shape
        self.use_se = False

        # AutoEncoder setting
        self.residul_latent_dim = 256  # dimension of latent vars. per group

        # encoder parameteres
        self.num_enc = args.num_enc  # each halfs the height and width
        self.in_chan_enc = args.in_chan_enc
        mult = 1
        self.encoder, mult = self.init_encoder(mult)

        # decoder parameters
        # number of cell for each conditional in decoder
        self.nf_dim_per_scale = 32

        # init prior and posterior
        self.init_pri_pos(args)

        self.generative_classifier = generative_classifier(
            self.nf_prior_z, args.attribute)

        # init decoder
        self.num_decoder = self.num_encoder
        self.decoder, mult = self.init_decoder(mult)

        # collect all norm params in Conv2D and gamma param in batchnorm
        self.all_log_norm = []
        self.all_conv_layers = []
        self.all_bn_layers = []
        for n, layer in self.named_modules():
            # if isinstance(layer, Conv2D) and '_ops' in n:   # only chose those in cell
            if isinstance(layer, Conv2D) or isinstance(layer, ARConv2d):
                self.all_log_norm.append(layer.log_weight_norm)
                self.all_conv_layers.append(layer)
            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.SyncBatchNorm) or \
                    isinstance(layer, SyncBatchNormSwish):
                self.all_bn_layers.append(layer)

        print('len log norm:', len(self.all_log_norm))
        print('len bn:', len(self.all_bn_layers))
        # left/right singular vectors used for SR
        self.sr_u = {}
        self.sr_v = {}
        self.num_power_iter = 4

    def init_encoder(self, mult):
        encoder = nn.ModuleList()
        for _ in range(self.num_deter_enc):
            num_ci = self.in_chan_deter_enc * mult
            cell = Cell(num_ci,
                        num_ci / 2,
                        cell_type='normal_pre',
                        arch=self.arch_instance['normal_pre'],
                        use_se=self.use_se)
            mult /= 2
            encoder.append(cell)
        return encoder, mult

    def init_pri_pos(self, args):
        in_chan_enc = self.in_chan_enc
        # half_in_chan_stoch_enc = in_chan_stoch_enc / 2
        self.posterior = Conv2D(in_chan_enc,
                                      2 * self.residul_latent_dim +
                                      2 * self.nf_dim_local,
                                      kernel_size=3,
                                      padding=1,
                                      bias=True)

        self.nf_prior_z = invertible_net(
            use_self_attn=args.use_self_attn,
            use_split=args.use_split,
            downsample=args.downsample,
            verbose=False,
            FlowBlocks_architecture=args.FlowBlocks_architecture,
            in_shape=self.nf_in_shape,
            in_shape_condition_node=in_shape_condition_node,
            mid_channels=args.num_channels,
            num_ConvAttnBlock=args.num_ConvAttnBlock,
            num_components=args.num_components,
            drop_prob=args.drop_prob)

        self.nf_prior_r = invertible_net(
            use_self_attn=args.use_self_attn,
            use_split=args.use_split,
            downsample=args.downsample,
            verbose=False,
            FlowBlocks_architecture=args.FlowBlocks_architecture,
            in_shape=self.nf_in_shape,
            in_shape_condition_node=None,
            mid_channels=args.num_channels,
            num_ConvAttnBlock=args.num_ConvAttnBlock,
            num_components=args.num_components,
            drop_prob=args.drop_prob,
            num_InvAutoFC=1)

    def init_nf_intervention(self, args):
        inn = invertible_net(
            use_self_attn=args.use_self_attn,
            use_split=args.use_split,
            downsample=args.downsample,
            verbose=False,
            FlowBlocks_architecture=args.FlowBlocks_architecture,
            in_shape=self.nf_in_shape,
            in_shape_condition_node=None,
            mid_channels=args.num_channels,
            num_ConvAttnBlock=args.num_ConvAttnBlock,
            num_components=args.num_components,
            drop_prob=args.drop_prob,
            num_InvAutoFC=1)

        self.nf_intervention =

    def init_decoder(self):
        decoder = nn.ModuleList()
        mult = 1
        for _ in range(self.num_decoder):
            num_ci = int(self.chan_in_deter_dec * mult)
            cell = Cell(num_ci,
                        num_ci * 2,
                        cell_type='normal_post',
                        arch=self.arch_instance['normal_post'],
                        use_se=self.use_se)
            mult *= 2
        decoder.append(cell)
        return decoder, mult

    def forward(self, x, one_hot_attributes, encoded_attributes):
        batch_size = x.size(0)
        # perform deterministic_encoder
        for cell in self.deterministic_encoder:
            s = cell(s)

        mu1, log_var1, mu2, log_var2 = torch.chunk(s, 4, dim=1)

        z_dis = Normal(mu1, log_var1)
        z_sample = z_dis.sample # [batch, 128, 7, 7]
        z_loss = self.nf_prior_z(z_sample, encoded_attributes)

        r_dis = Normal(mu2, log_var2)
        r_sample = r_dis.sample
        r_loss = self.nf_prior_r(r_sample, encoded_attributes)

        observed 
        intervented 
        intervented = self.nf_intervention(z_sample, one_hot_attributes)

        # concat as short-cut
        s = torch.cat(z_sample, r_sample, dim=1)

        # perform deterministic_decoder
        for cell in self.deterministic_decoder:
            s = cell(s)

            rec_loss = torch.sum(torch.abs(x - s), dim=(1, 2, 3)) / batch_size

        return rec_loss, z_loss1, z_loss2

    def sample(self, num_samples, t):
        scale_ind = 0
        z0_size = [num_samples] + self.z0_size
        dist = Normal(mu=torch.zeros(z0_size).cuda(),
                      log_sigma=torch.zeros(z0_size).cuda(),
                      temp=t)
        z, _ = dist.sample()

        idx_dec = 0
        s = self.prior_ftr0.unsqueeze(0)
        s = s.expand(self.batch_size, -1, -1, -1)
        for cell in self.stochastic_decoder:
            if cell.cell_type == 'combiner_dec':
                if idx_dec > 0:
                    # form prior
                    param = self.prior_sampler[idx_dec - 1](s)
                    mu, log_sigma = torch.chunk(param, 2, dim=1)
                    dist = Normal(mu, log_sigma, t)
                    z, _ = dist.sample()

                # 'combiner_dec'
                s = cell(s, z)
                idx_dec += 1
            else:
                s = cell(s)
                if cell.cell_type == 'up_dec':
                    scale_ind += 1

        for cell in self.deterministic_decoder:
            s = cell(s)

        logits = self.image_conditional(s)
        return logits

    def spectral_norm_parallel(self):
        """ This method computes spectral normalization for all conv layers in parallel. This method should be called after calling the forward method of all the conv layers in each iteration. """

        weights = {}  # a dictionary indexed by the shape of weights
        for l in self.all_conv_layers:
            weight = l.weight_normalized
            weight_mat = weight.view(weight.size(0), -1)
            if weight_mat.shape not in weights:
                weights[weight_mat.shape] = []

            weights[weight_mat.shape].append(weight_mat)

        loss = 0
        for i in weights:
            weights[i] = torch.stack(weights[i], dim=0)
            with torch.no_grad():
                num_iter = self.num_power_iter
                if i not in self.sr_u:
                    num_w, row, col = weights[i].shape
                    self.sr_u[i] = F.normalize(torch.ones(num_w, row).normal_(
                        0, 1).cuda(),
                                               dim=1,
                                               eps=1e-3)
                    self.sr_v[i] = F.normalize(torch.ones(num_w, col).normal_(
                        0, 1).cuda(),
                                               dim=1,
                                               eps=1e-3)
                    # increase the number of iterations for the first time
                    num_iter = 10 * self.num_power_iter

                for j in range(num_iter):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    self.sr_v[i] = F.normalize(
                        torch.matmul(self.sr_u[i].unsqueeze(1),
                                     weights[i]).squeeze(1),
                        dim=1,
                        eps=1e-3)  # bx1xr * bxrxc --> bx1xc --> bxc
                    self.sr_u[i] = F.normalize(
                        torch.matmul(weights[i],
                                     self.sr_v[i].unsqueeze(2)).squeeze(2),
                        dim=1,
                        eps=1e-3)  # bxrxc * bxcx1 --> bxrx1  --> bxr

            sigma = torch.matmul(
                self.sr_u[i].unsqueeze(1),
                torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)))
            loss += torch.sum(sigma)
        return loss

    def batchnorm_loss(self):
        loss = 0
        for l in self.all_bn_layers:
            if l.affine:
                loss += torch.max(torch.abs(l.weight))
        return loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        nn.init.xavier_uniform_(m.weight, gain=0.5)


class Attr_AutoEncoder(nn.Module):
    def __init__(self, dim_attribute, dim_interme_layer, dim_latent_layer):
        super(Attr_AutoEncoder, self).__init__()
        self.dim_latent_layer = dim_latent_layer
        self.encoder = nn.Sequential(
            nn.Linear(dim_attribute, dim_interme_layer),
            nn.ReLU(),
            nn.Linear(dim_interme_layer, dim_latent_layer),
            nn.ReLU(),
            nn.Linear(dim_latent_layer, dim_latent_layer * 2),
        )
        self.apply(weights_init)
        self.decoder = nn.Sequential(
            nn.Linear(dim_latent_layer, dim_interme_layer), nn.ReLU(),
            nn.Linear(dim_interme_layer, dim_attribute))

    def forward(self, x):
        mu, log_sigma = torch.chunk(self.encoder(x), 2, dim=1)
        dist = Normal(mu, log_sigma)
        kl_loss = dist.kl(
            Normal(
                torch.zeros(self.dim_latent_layer).cuda(),
                torch.ones(self.dim_latent_layer).cuda()))
        reconstructed = self.decoder(dist.sample())
        recon_loss = torch.abs(reconstructed, x)

        return recon_loss, kl_loss
