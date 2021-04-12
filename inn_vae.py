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
from inn_model import inn_classifier, inn_intervention
from models.flow.invertible_net import invertible_net
import utils
from models.flow.flowpp_coupling import GatedAttn
import random
from pl_bolts.models.self_supervised import CPCV2

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
    def __init__(self, args, arch_instance):
        super(inn_vae, self).__init__()
        self.in_shape = args.in_shape
        # encoder parameteres# each halfs the height and width
        self.in_chan_enc = args.in_chan_enc
        self.encoder = self.init_encoder()
        self.use_self_attn = 1
        # decoder parameters
        # number of cell for each conditional in decoder
        self.inn_dim_per_scale = 32

        # init prior and posterior
        self.init_prior(args)
        self.inn_classifier = inn_classifier()

        # init decoder
        self.num_decoder = self.num_encoder
        self.decoder = self.init_decoder()

        weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpcv2_weights/checkpoints/epoch%3D526.ckpt'
        self.backbone = CPCV2.load_from_checkpoint(weight_path, strict=False)

        # # collect all norm params in Conv2D and gamma param in batchnorm
        # self.all_log_norm = []
        # self.all_conv_layers = []
        # self.all_bn_layers = []

        # for n, layer in self.named_modules():
        #     # if isinstance(layer, Conv2D) and '_ops' in n:   # only chose those in cell
        #     if isinstance(layer, Conv2D) or isinstance(layer, ARConv2d):
        #         self.all_log_norm.append(layer.log_weight_norm)
        #         self.all_conv_layers.append(layer)
        #     if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.SyncBatchNorm) or \
        #             isinstance(layer, SyncBatchNormSwish):
        #         self.all_bn_layers.append(layer)

    def init_encoder(self):
        deterministic_encoder = nn.ModuleList()
        num_ci = self.in_chan_deter_enc
        for _ in range(self.num_deter_enc):
            conv = Conv2D(num_ci,
                        num_ci / 2,
                        kernel_size=3,
                        padding=1,
                        bias=True)
            num_ci /= 2
            deterministic_encoder.append(conv)
        return deterministic_encoder

    def init_prior(self, args):
        self.inn_prior_z = invertible_net(
            use_self_attn=args.use_self_attn,
            use_split=args.use_split,
            downsample=args.downsample,
            verbose=False,
            FlowBlocks_architecture=args.FlowBlocks_architecture,
            in_shape=self.inn_in_shape,
            in_shape_condition_node=args.in_shape_condition_node,
            mid_channels=args.num_channels,
            num_ConvAttnBlock=args.num_ConvAttnBlock,
            num_components=args.num_components,
            drop_prob=args.drop_prob)

        if self.c_use_nf_pri:
            self.inn_prior_c = invertible_net(
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

        if self.r_use_nf_pri:
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


    def init_decoder(self):
        decoder = nn.ModuleList()
        num_ci = self.chan_in_deter_dec
        for _ in range(self.num_decoder):
            if self.use_self_attn:
                attn = GatedAttn(num_ci, num_heads=4, drop_prob=0)
                decoder.append(attn)

            conv = Conv2D(
                        num_ci,
                        num_ci * 2,
                        kernel_size=3,
                        padding=1,
                        bias=True)
            num_ci *= 2
            decoder.append(conv)
        return decoder

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

    def intervention_on_patch(self, label, intervented_patch, embedings_after_intervention):
        f_z = self.inn_prior_z(self.z_sample, c=intervented_patch)
        log_jacobian_z = self.inn_prior_z.log_jacobian(run_forward=False)
        log_p_f_z, cross_entropy = self.inn_classifier(f_z, label, embedings_after_intervention)

        return log_p_f_z, log_jacobian_z, cross_entropy
    
    def decoder(self, input):
        for decoder in self.decoder:
            input = decoder(input)
        return input

    def intervention_on_z(self):
        # f_z_sample = self.inn_classifier.sample(labels)
        shuffle = random.shuffle(list(range(self.batch_size))) 
        shuffled_z_sample = self.z_sample[shuffle, :, :, :]

        w_sample = torch.cat(shuffled_z_sample, self.r_sample)
        decodered = self.decoder(w_sample)
        
        feature_map = torch.mean(decodered, dim=[2, 3])

        return feature_map

    def intervention_on_c(self):
        shuffle = random.shuffle(list(range(self.batch_size))) 
        shuffled_r_sample = self.r_sample[shuffle, :, :, :]

        w_sample = torch.cat(shuffled_r_sample, self.r_sample)
        decodered = self.decoder(w_sample)

        feature_map = torch.mean(decodered, dim=[2, 3])

        return feature_map

    def forward(self, inputs, label, num_train):

        with torch.no_grad():
            feature_map = self.backbone(inputs)

        [batch_size, channels, height, weight] = feature_map.size()
        # perform encoder
        for encoder in self.encoder:
            inputs = encoder(inputs)

        mu1, log_var1, mu2, log_var2 = torch.chunk(inputs, 4, dim=1)
        # for semantic discrimitive Z
        self.z_dis = Normal(mu1, log_var1)
        self.z_sample = self.z_dis.sample # [batch, 128, 7, 7]

        f_z = self.inn_prior_z(self.z_sample)
        log_jacobian_z = self.inn_prior_z.log_jacobian(run_forward=False)
        log_p_f_z, cross_entropy = self.inn_classifier(f_z, label) # [batch, -1]
        log_q_z = self.z_dis.log_p(self.z_sample)

        regul_z = log_p_f_z + log_jacobian_z - \
            torch.log(self.entropy_estimator(log_q_z, self.num_train))

        # residul non-semantic non-discrimitive r contains other information for reconstruction
        # dis_z_a = self.inn_classifier.cluster_distances(f_z, encoded_attributes)
        # deter_z = self.inn_prior_z.log_jacobian_numerical #todo
        self.r_dis = Normal(mu2, log_var2)
        self.r_sample = self.r_dis.sample

        if self.r_use_nf_pri == True:
            f_r = self.inn_prior_r(self.r_sample)
            log_jacobian_r = self.inn_prior_r.log_jacobian(run_forward=False)
            normal_r = Normal(torch.zeros_like(f_r), torch.ones_like(f_r))
        else:
            # f is identical function
            f_r = self.r_sample
            log_jacobian_r = 0
            normal_r = Normal(torch.zeros_like(f_r), torch.ones_like(f_r))

        log_p_f_r = normal_r.log_p(f_r)
        log_q_r = self.r_dis.log_p(self.r_sample)

        regul_r = log_p_f_r + log_jacobian_r - torch.log(
            self.entropy_estimator(log_q_r, num_train))

        # for w
        log_q_w = torch.cat(log_q_z, log_q_r, dim=1)
        w_sample = torch.cat(self.z_sample, self.r_sample)
        # perform decoder
        decodered = self.decoder(w_sample)

        mutual_info_wx = torch.reshape(log_q_w, [batch_size, -1]) \
            - torch.log(self.entropy_estimator(log_q_w, num_train))

        # todo testing L_2 distance
        # here we use cosine_similarity which is bounded between [-1, 1] to ensure recon_loss in the same magnitude to other terms which are normalized.
        recon = torch.cosine_similarity(
            torch.reshape(inputs, [batch_size, -1]), 
            torch.reshape(decodered, [batch_size, -1]))
        # channelwise_recon_loss = torch.cosine_similarity(
        #     torch.reshape(inputs, [batch_size, channels, -1]), 
        #     torch.reshape(decodered, [batch_size, channels, -1]), dim=1)

        # perform rebalancing 
        regul_z = torch.mean(regul_z, dim=[1]) 
        regul_r = torch.mean(regul_r, dim=[1]) 
        mutual_info_wx = torch.mean(mutual_info_wx, dim=[1]) 
        recon = torch.mean(recon, dim=[1]) 

        return recon, regul_z, regul_r, cross_entropy, mutual_info_wx

    def counterfacture(self,):
        pass

    # def spectral_norm_parallel(self):
    #     """ This method computes spectral normalization for all conv layers in parallel. This method should be called after calling the forward method of all the conv layers in each iteration. """

    #     weights = {}  # a dictionary indexed by the shape of weights
    #     for l in self.all_conv_layers:
    #         weight = l.weight_normalized
    #         weight_mat = weight.view(weight.size(0), -1)
    #         if weight_mat.shape not in weights:
    #             weights[weight_mat.shape] = []

    #         weights[weight_mat.shape].append(weight_mat)

    #     loss = 0
    #     for i in weights:
    #         weights[i] = torch.stack(weights[i], dim=0)
    #         with torch.no_grad():
    #             num_iter = self.num_power_iter
    #             if i not in self.sr_u:
    #                 num_w, row, col = weights[i].shape
    #                 self.sr_u[i] = F.normalize(torch.ones(num_w, row).normal_(
    #                     0, 1).cuda(),
    #                                            dim=1,
    #                                            eps=1e-3)
    #                 self.sr_v[i] = F.normalize(torch.ones(num_w, col).normal_(
    #                     0, 1).cuda(),
    #                                            dim=1,
    #                                            eps=1e-3)
    #                 # increase the number of iterations for the first time
    #                 num_iter = 10 * self.num_power_iter

    #             for j in range(num_iter):
    #                 # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
    #                 # are the first left and right singular vectors.
    #                 # This power iteration produces approximations of `u` and `v`.
    #                 self.sr_v[i] = F.normalize(
    #                     torch.matmul(self.sr_u[i].unsqueeze(1),
    #                                  weights[i]).squeeze(1),
    #                     dim=1,
    #                     eps=1e-3)  # bx1xr * bxrxc --> bx1xc --> bxc
    #                 self.sr_u[i] = F.normalize(
    #                     torch.matmul(weights[i],
    #                                  self.sr_v[i].unsqueeze(2)).squeeze(2),
    #                     dim=1,
    #                     eps=1e-3)  # bxrxc * bxcx1 --> bxrx1  --> bxr

    #         sigma = torch.matmul(
    #             self.sr_u[i].unsqueeze(1),
    #             torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)))
    #         loss += torch.sum(sigma)
    #     return loss

    # def batchnorm_loss(self):
    #     loss = 0
    #     for l in self.all_bn_layers:
    #         if l.affine:
    #             loss += torch.max(torch.abs(l.weight))
    #     return loss


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