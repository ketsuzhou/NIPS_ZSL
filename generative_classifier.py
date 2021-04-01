from functools import partial
import pdb

import torch
import torch.nn as nn
import numpy as np

from models.flow.invertible_net import *

class nf_classifier(nn.Module):
    def __init__(self, condition_encoder, inn_prior_sampler, attribute):
        super(nf_classifier, self).__init__()
        # self.args = args
        # self.invertible_nets = nn.ModuleList()
        attribute_transductive = 1
        self.dim_attribute = len(attribute)
        self.num_inn = len(inn_prior_sampler)
        # self.num_inn = num_latent_scales * num_groups_per_scale

        self.dim_attribute_per_inn = self.dim_attribute / self.num_inn

        self.num_classes = int(attribute.shape[0])
        self.num_dim = int(attribute.size(1))
        mu_populate_dims = self.num_dim
        self.mu = nn.Parameter(torch.zeros(1, self.num_classes, mu_populate_dims))
        init_latent_scale = 5.0
        init_scale = init_latent_scale / np.sqrt(2 * mu_populate_dims // self.num_classes)

        for k in range(mu_populate_dims // self.num_classes):
            self.mu.data[0, :, self.num_classes * k : self.num_classes * (k+1)] = init_scale * torch.eye(self.num_classes)
        
        self.phi = nn.Parameter(torch.zeros(self.num_classes))

        self.trainable_params = list(self.invertible_net.parameters())
        self.trainable_params = list(filter((lambda p: p.requires_grad), self.trainable_params))
        weight_init = 1.0
        for p in self.trainable_params:
            p.data *= weight_init

        self.train_inn = True

        # self.trainable_params += [self.mu, self.phi]
        self.trainable_params += [self.phi]
        base_lr = float(self.args.lr)

        optimizer_params = [{
            'params':list(filter(lambda p: p.requires_grad, self.invertible_net.parameters()))
            },]
        
        optimizer = self.args.train['optimizer']
        if optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(optimizer_params, base_lr,
                                              momentum=float(self.args['training']['sgd_momentum']),
                                              weight_decay=float(self.args['training']['weight_decay']))
        elif optimizer == 'ADAM':
            self.optimizer = torch.optim.Adam(optimizer_params, base_lr,
                                              betas=self.args.train['adam_betas'],
                                              weight_decay=float(self.args.train['weight_decay']))
        elif optimizer == 'AGGMO':
            import aggmo
            self.optimizer = aggmo.AggMo(optimizer_params, base_lr,
                                              betas=eval(self.args['training']['aggmo_betas']),
                                              weight_decay=float(self.args['training']['weight_decay']))
        else:
            raise ValueError(f'what is this optimizer, {optimizer}?')

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

    def mu_pairwise_dist(self):

        mu_i_mu_j = self.mu.squeeze().mm(self.mu.squeeze().t())
        mu_i_mu_i = torch.sum(self.mu.squeeze()**2, 1, keepdim=True).expand(self.n_classes, self.n_classes)

        dist =  mu_i_mu_i + mu_i_mu_i.t() - 2 * mu_i_mu_j
        return torch.masked_select(dist, (1 - torch.eye(self.n_classes).cuda()).bool()).clamp(min=0.)

    def forward(self, x, y=None, loss_mean=True):

        if self.feed_forward:
            return self.losses_feed_forward(x, y, loss_mean)

        z = self.invertible_net(x)
        jac = self.invertible_net.log_jacobian(run_forward=False)

        log_wy = torch.log_softmax(self.phi, dim=0).view(1, -1)
        zz = self.cluster_distances(z, y)
        losses = {
            'L_x_tr': (- torch.logsumexp(- 0.5 * zz + log_wy, dim=1) - jac ) / self.ndim_tot,  'logits_tr': - 0.5 * zz}
        # detach log_wy to block gradient
        log_wy = log_wy.detach()
        if y is not None:
            # losses['L_cNLL_tr'] = (0.5 * torch.sum(zz * y.round(), dim=1) - jac) / self.ndim_tot
            losses['L_y_tr'] = torch.sum(
                (torch.log_softmax(- 0.5 * zz + log_wy, dim=1) - log_wy) * y, dim=1)
            
            losses['acc_tr'] = torch.mean(
                (torch.max(y, dim=1)[1]  
                == torch.max(losses['logits_tr'].detach(), dim=1)[1]).float())

        if loss_mean:
            for k, v in losses.items():
                losses[k] = torch.mean(v)

        return losses

    def losses_feed_forward(self, x, y=None, loss_mean=True):
        logits = self.invertible_net(x)

        losses = {'logits_tr': logits,
                  'L_x_tr': torch.zeros_like(logits[:,0])}

        if y is not None:
            ly =  torch.sum(torch.log_softmax(logits, dim=1) * y, dim=1)
            acc = torch.mean((torch.max(y, dim=1)[1]
                           == torch.max(logits.detach(), dim=1)[1]).float())
            losses['L_y_tr'] = ly
            losses['acc_tr'] = acc
            losses['L_cNLL_tr'] = torch.zeros_like(ly)

        if loss_mean:
            for k,v in losses.items():
                losses[k] = torch.mean(v)

        return losses

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

    def reset_mu(self, dataset):
        mu = torch.zeros(1, self.n_classes, self.ndim_tot).cuda()
        counter = 0

        with torch.no_grad():
            for x, l in dataset.train_loader:
                x, y = x.cuda(), dataset.onehot(l.cuda(), 0.05)
                z = self.invertible_net(x)
                mu_batch = torch.mm(z.t().detach(), y.round())
                mu_batch = mu_batch / torch.sum(y, dim=0, keepdim=True)
                mu_batch = mu_batch.t().view(1, self.n_classes, -1)

                mu += mu_batch
                counter += 1

            mu /= counter
        self.mu.data  = mu.data

    def sample(self, y, temperature=1.):
        z = temperature * torch.randn(y.shape[0], self.ndim_tot).cuda()
        mu = torch.sum(y.round().view(-1, self.n_classes, 1) * self.mu, dim=1)
        return self.invertible_net(z, rev=True)

    def save(self, fname):
        torch.save({'invertible_net': self.invertible_net.state_dict(),
                    'mu':  self.mu,
                    'phi': self.phi,
                    'opt': self.optimizer.state_dict()}, fname)

    def load(self, fname):
        data = torch.load(fname)
        data['invertible_net'] = {k:v for k,v in data['invertible_net'].items() if 'tmp_var' not in k}
        self.invertible_net.load_state_dict(data['invertible_net'])
        self.mu.data.copy_(data['mu'].data)
        self.phi.data.copy_(data['phi'].data)
        try:
            pass
        except:
            print('loading the optimizer went wrong, skipping')