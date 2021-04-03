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
from inn_interv_classi import inn_classifier, inn_intervention
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

def cluster_distances(self, z, y=None):

    if y is not None:
        mu = torch.mm(z.t().detach(), y.round())
        mu = mu / torch.sum(y, dim=0, keepdim=True)
        mu = mu.t().view(1, self.n_classes, -1)
        mu = 0.005 * mu + 0.995 * self.mu.data
        self.mu.data = mu.data

    z_i_z_i = torch.sum(z**2, dim=1, keepdim=True)   # batchsize x n_classes
    mu_j_mu_j = torch.sum(self.mu**2, dim=2)         # 1 x n_classes
    z_i_mu_j = torch.mm(z, self.mu.squeeze().t())    # batchsize x n_classes

    return -2 * z_i_mu_j + z_i_z_i + mu_j_mu_j
        

class inn_vae(nn.Module):
    def __init__(self, args, writer, arch_instance):
        super(inn_vae, self).__init__()
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
        self.inn_dim_per_scale = 32

        # init prior and posterior
        self.init_pri_pos(args)

        # self.inn_prior_z = inn_classifier(
        #     self.inn_prior_z, args.attribute)
        self.inn_classifier = inn_classifier()
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
                                      2 * self.inn_dim_local,
                                      kernel_size=3,
                                      padding=1,
                                      bias=True)

        self.inn_prior_z = invertible_net(
            use_self_attn=args.use_self_attn,
            use_split=args.use_split,
            downsample=args.downsample,
            verbose=False,
            FlowBlocks_architecture=args.FlowBlocks_architecture,
            in_shape=self.inn_in_shape,
            in_shape_condition_node=in_shape_condition_node,
            mid_channels=args.num_channels,
            num_ConvAttnBlock=args.num_ConvAttnBlock,
            num_components=args.num_components,
            drop_prob=args.drop_prob)

        self.inn_prior_r = invertible_net(
            use_self_attn=args.use_self_attn,
            use_split=args.use_split,
            downsample=args.downsample,
            verbose=False,
            FlowBlocks_architecture=args.FlowBlocks_architecture,
            in_shape=self.inn_in_shape,
            in_shape_condition_node=None,
            mid_channels=args.num_channels,
            num_ConvAttnBlock=args.num_ConvAttnBlock,
            num_components=args.num_components,
            drop_prob=args.drop_prob,
            num_InvAutoFC=1)

    def init_inn_intervention(self, args):
        inn = invertible_net(
            use_self_attn=args.use_self_attn,
            use_split=args.use_split,
            downsample=args.downsample,
            verbose=False,
            FlowBlocks_architecture=args.FlowBlocks_architecture,
            in_shape=self.inn_in_shape,
            in_shape_condition_node=None,
            mid_channels=args.num_channels,
            num_ConvAttnBlock=args.num_ConvAttnBlock,
            num_components=args.num_components,
            drop_prob=args.drop_prob,
            num_InvAutoFC=1)

        self.inn_intervention = inn_intervention(inn )

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

    def entropy_estimator(self, log_q_latent_condi_x, num_train):
        q_latent_condi_x = torch.exp(log_q_latent_condi_x)
        sum_q_over_remaining = torch.sum(q_latent_condi_x, dim=0, keepdim=True)
        batch_size = log_q_latent_condi_x.size(0)

        for i in range(batch_size):
            sum_q_over_remaining[i, :, :, :] -= q_latent_condi_x[i, :, :, :]
         
        q_latent = (1 / num_train )* q_latent_condi_x + \
            ((num_train - 1) / (num_train * (batch_size - 1))) * sum_q_over_remaining
        # upper bound of entropy for approximation
        # entropy_q_latent = torch.mean(torch.reshape(torch.log(q_latent), [batch_size, -1]) , dim=0)  
        # calculating mean over batch as above resulting entropy_q_latent
        log_q_latent = torch.reshape(torch.log(q_latent), [batch_size, -1]) 
        return log_q_latent

    def intervention(self ):
        z_sample = self.z_dis.sample # [batch, 128, 7, 7]
        intervention_loss = self.inn_intervention(z_sample)
        return intervention_loss

    def forward(self, inputs, encoded_attributes, num_train):
        [batch_size, channels, height, weight] = inputs.size()
        # perform encoder
        for encoder in self.encoder:
            inputs = encoder(inputs)

        mu1, log_var1, mu2, log_var2 = torch.chunk(inputs, 4, dim=1)

        self.z_dis = Normal(mu1, log_var1)
        z_sample = self.z_dis.sample # [batch, 128, 7, 7]
        f_z = self.inn_prior_z(z_sample)
        log_jacobian_z = self.inn_prior_z.log_jacobian(run_forward=False)

        log_p_f_z, bayes_loss = self.inn_classifier(f_z, y) # [batch, -1]

        log_q_z = self.z_dis.log_p(z_sample)

        regul_z = log_p_f_z + log_jacobian_z - \
            torch.log(self.entropy_estimator(log_q_z, num_train))

        # dis_z_a = self.inn_classifier.cluster_distances(f_z, encoded_attributes)

        # deter_z = self.inn_prior_z.log_jacobian_numerical #todo
        self.r_dis = Normal(mu2, log_var2)
        r_sample = self.r_dis.sample
        f_r = self.inn_prior_r(r_sample)
        normal = Normal(torch.zeros_like(f_r), torch.ones_like(f_r))
        log_jacobian_r = self.inn_prior_r.log_jacobian(run_forward=False)
        
        log_p_f_r = normal.log_p(f_r)
        log_q_r = self.r_dis.log_p(r_sample)

        log_q_w = torch.cat(log_q_z, log_q_r, dim=1)

        w_sample = torch.cat(z_sample, r_sample)

        decodered = w_sample
        # perform decoder
        for decoder in self.decoder:
            decodered = decoder(decodered)
        
        # todo testing L_2 distance
        recon_loss = torch.cosine_similarity(
            torch.reshape(inputs, [batch_size, -1]), 
            torch.reshape(decodered, [batch_size, -1]))
        # channelwise_recon_loss = torch.cosine_similarity(
        #     torch.reshape(inputs, [batch_size, channels, -1]), 
        #     torch.reshape(decodered, [batch_size, channels, -1]), dim=1)

        regul_r = log_p_f_r + log_jacobian_r - torch.log(
            self.entropy_estimator(log_q_r, num_train))
        
        # 
        mutual_info_wx = torch.reshape(log_q_w, [batch_size, -1]) \
             - torch.log(self.entropy_estimator(log_q_w, num_train))

        return recon_loss, regul_z, regul_r, bayes_loss, mutual_info_wx

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
