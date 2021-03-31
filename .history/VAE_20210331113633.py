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


class AutoEncoder(nn.Module):
    def __init__(self, args, writer, arch_instance):
        super(AutoEncoder, self).__init__()
        # self.writer = writer
        self.arch_instance = arch_instance
        self.in_shape = args.in_shape
        self.use_se = False
        # self.res_dist = args.res_dist        # todo

        # AutoEncoder setting
        self.num_latent_scales = 2         # number of spatial scales that latent layers will reside
        self.num_groups_per_scale = 1  # number of groups of latent vars. per scale default 64
        self.residul_latent_dim = 256  # dimension of latent vars. per group
        self.groups_per_scale = 1

        # encoder parameteres
        self.num_deter_enc = args.num_deter_enc  # each halfs the height and width
        self.in_chan_deter_enc = args.in_chan_deter_enc
        mult = 1
        self.num_cell_per_cond_enc = args.num_cell_per_cond_enc
        self.deterministic_encoder, mult = self.init_deterministic_encoder(mult)
        
        self.in_chan_stoch_enc = mult * self.in_chan_deter_enc
        self.stochastic_encoder = self.init_stochastic_encoder()

        # decoder parameters
        # number of cell for each conditional in decoder
        self.num_cell_per_cond_dec = args.num_cell_per_cond_dec
        self.nf_dim_per_scale = 32

        # init prior and posterior
        self.in_chan_condition = args.in_chan_condition
        self.norm_prior_sampler, self.condition_encoder, \
            self.inn_prior_sampler, self.posterior_sampler = self.init_pri_pos_sampler(args)

        self.generative_classifier = generative_classifier(
            self.condition_encoder, self.inn_prior_sampler, args.attribute)

        # init decoder
        self.num_deterministic_decoder = self.num_deterministic_encoder
        self.stochastic_decoder, mult = self.init_stochastic_decoder(mult)
        self.deterministic_decoder, mult = self.init_deterministic_decoder(mult)

        self.image_conditional = self.init_image_conditional(mult)

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

    def init_deterministic_encoder(self, mult):
        deterministic_encoder = nn.ModuleList()
        for _ in range(self.num_deter_enc):
            num_ci = self.in_chan_deter_enc * mult
            cell = Cell(num_ci,
                        num_ci / 2,
                        cell_type='normal_pre',
                        arch=self.arch_instance['normal_pre'],
                        use_se=self.use_se)
            mult /= 2
            deterministic_encoder.append(cell)
        return deterministic_encoder, mult


    def init_stochastic_encoder(self):
        in_chan_stoch_enc = self.in_chan_stoch_enc
        half_in_chan_stoch_enc = in_chan_stoch_enc / 2
        self.combiner_enc = EncCombinerCell(
            half_in_chan_stoch_enc *2,
            half_in_chan_stoch_enc,
            cell_type='combiner_enc')
        self.down1 = Cell(
            in_chan_stoch_enc,
            half_in_chan_stoch_enc,
            cell_type='down_enc',
            arch=self.arch_instance['down_enc'],
            use_se=self.use_se)
        self.enc = Cell(
            half_in_chan_stoch_enc,
            half_in_chan_stoch_enc,
            cell_type='normal_enc',
            arch=self.arch_instance['normal_enc'],
            use_se=self.use_se)
        self.down2 = Cell(
            half_in_chan_stoch_enc,
            half_in_chan_stoch_enc,
            cell_type='down_enc',
            arch=self.arch_instance['down_enc'],
            use_se=self.use_se)


    def init_pri_pos_sampler(self, args):
        in_chan_stoch_enc = self.in_chan_stoch_enc
        half_in_chan_stoch_enc = in_chan_stoch_enc / 2
        self.local_posterior = Conv2D(
            in_chan_stoch_enc,
            2 * self.residul_latent_dim + 2 * self.nf_dim_local, 
            kernel_size=3, 
            padding=1, 
            bias=True)
        # condition for local NF prior
        self.condition_encoder = nn.ModuleList(
            nn.Sequential(
                nn.ELU(),
                Conv2D(
                    half_in_chan_stoch_enc,
                    self.in_chan_condition,
                    kernel_size=1,
                    padding=0,
                    bias=True)))
        self.local_nf_prior_z = invertible_net(
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
        self.global_posterior = Conv2D(
            in_chan_stoch_enc,
            2 * self.nf_dim_nonlocal, 
            kernel_size=3, 
            padding=1, 
            bias=True)
        self.global_nf_prior = invertible_net(
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


    def init_stochastic_decoder(self):
        half_in_chan_stoch_enc = self.in_chan_stoch_enc / 2
        self.up1 = Cell(
            half_in_chan_stoch_enc,
            half_in_chan_stoch_enc,
            cell_type='up_dec',
            arch=self.arch_instance['up_dec'],
            use_se=self.use_se)
        self.dec = Cell(
            half_in_chan_stoch_enc,
            half_in_chan_stoch_enc,
            cell_type='normal_dec',
            arch=self.arch_instance['normal_dec'],
            use_se=self.use_se)
        self.up2 = Cell(
            half_in_chan_stoch_enc,
            half_in_chan_stoch_enc,
            cell_type='up_dec',
            arch=self.arch_instance['up_dec'],
            use_se=self.use_se)
        self.combiner_dec = DecCombinerCell(
            2 * half_in_chan_stoch_enc,
            half_in_chan_stoch_enc,
            half_in_chan_stoch_enc,
            cell_type='combiner_dec')


    def init_deterministic_decoder(self):
        deterministic_decoder = nn.ModuleList()
        mult = 1
        for _ in range(self.num_deterministic_decoder):
            num_ci = int(self.chan_in_deter_dec * mult)
            cell = Cell(num_ci,
                        num_ci * 2,
                        cell_type='normal_post',
                        arch=self.arch_instance['normal_post'],
                        use_se=self.use_se)
            mult *= 2
        deterministic_decoder.append(cell)
        return deterministic_decoder, mult

    def forward(self, x):
        batch_size = x.size(0)
        # perform deterministic_encoder
        for cell in self.deterministic_encoder:
            s = cell(s)
        # save encoded_feature
        encoded_feature = s
        # down sample
        s = self.down1(s)
        s = self.enc(s)
        s = self.down2(s)
        mu, log_var = torch.chunk(s, 2, dim=1)
        z_global_dis = Normal(mu, log_var)
        z_global_sample = z_global_dis.sample
        # global_nf_prior
        z_loss1 = self.global_nf_prior(z_global_sample)
        # decode
        s = self.up1(z_global_sample)
        s = self.dec(s)
        decoded_feature = self.up2(s)
        s = self.combiner_enc(encoded_feature, decoded_feature)
        mu1, log_var1, mu2, log_var2 = torch.chunk(s, 4, dim=1)
        z_local = Normal(mu1, log_var1)
        z_local_sample = z_local.sample
        # local_nf_prior
        local_nf_condition = self.condition_encoder(decoded_feature)
        z_loss2 = self.local_nf_prior_z(local_nf_condition, z_local_sample)
        # r prior
        r = Normal(mu2, log_var2)
        # concat as short-cut
        s = torch.cat(decoded_feature, z_local, r, dim=1)

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
                        0, 1).cuda(), dim=1, eps=1e-3)
                    self.sr_v[i] = F.normalize(torch.ones(num_w, col).normal_(
                        0, 1).cuda(), dim=1, eps=1e-3)
                    # increase the number of iterations for the first time
                    num_iter = 10 * self.num_power_iter

                for j in range(num_iter):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    self.sr_v[i] = F.normalize(
                        torch.matmul(self.sr_u[i].unsqueeze(1),
                                     weights[i]).squeeze(1), dim=1, eps=1e-3)  # bx1xr * bxrxc --> bx1xc --> bxc
                    self.sr_u[i] = F.normalize(
                        torch.matmul(weights[i],
                                     self.sr_v[i].unsqueeze(2)).squeeze(2), dim=1, eps=1e-3)  # bxrxc * bxcx1 --> bxrx1  --> bxr

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
