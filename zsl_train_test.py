import matplotlib.pyplot as plt
from bagnets.utils import plot_heatmap, generate_heatmap_pytorch
import bagnets.pytorchnet
import argparse
import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import util
from data_factory.dataloader import *
from arg_completion import *
from data_factory.data_transform import TrainTransforms
from models.flow.invertible_net import *
from tqdm import tqdm
import yaml
from configs.default import cfg, update_datasets
from inn_model import inn_classifier
from data_module import data_module
from data_factory.dataloader import IMAGE_LOADOR
from pl_bolts.models.self_supervised import CPCV2, SSLFineTuner
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import utils
from torch.multiprocessing import Process
import torch.distributed as dist
from inn_vae import inn_vae
from adamax import Adamax
from torch.cuda.amp import autocast, GradScaler
from init_saving import create_filename
import logging
logger = logging.getLogger(__name__)


def train_test(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get data loaders.
    dm = data_module(args)
    if args.zsl_type == "conventional":
        train_dataloader, test_dataloader = dm.conventional_dataloader()
    elif args.zsl_type == "generalized":
        train_dataloader, test_dataloader = dm.generalized_dataloader()
    else:
        NotImplementedError

    # # load model
    # pytorch_model = bagnets.pytorchnet.bagnet33(pretrained=True).cuda()
    # pytorch_model.eval()

    # for i, (X, label) in enumerate(train_dataloader):
    #     original = X.cuda()
    #     sample = X / 255.
    #     sample -= np.array([0.485, 0.456, 0.406])[:, None, None]
    #     sample /= np.array([0.229, 0.224, 0.225])[:, None, None]

    #     # generate heatmap
    #     heatmap = generate_heatmap_pytorch(pytorch_model, sample, label, 33)

    #     # plot heatmap
    #     fig = plt.figure(figsize=(8, 4))
    #     original_image = original[0].permute(1, 2, 0).cpu()
    #     # original_image = original[0].transpose(1, 2, 0)

    #     ax = plt.subplot(121)
    #     ax.set_title('original')
    #     plt.imshow(original_image / 255.)
    #     plt.axis('off')

    #     ax = plt.subplot(122)
    #     ax.set_title('heatmap')
    #     plot_heatmap(heatmap, original_image, ax,
    #                  dilation=0.5, percentile=99, alpha=.25)
    #     plt.axis('off')

    #     plt.show()

    args.num_total_iter = len(train_dataloader) * args.epochs
    warmup_iters = len(train_dataloader) * args.warmup_epochs
    swa_start = len(train_dataloader) * (args.epochs - 1)

    if args.backbone == 'resnet101':
        backbone = models.__dict__['resnet101'](pretrained=True)
    else:
        weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpcv2_weights/checkpoints/epoch%3D526.ckpt'
        backbone = CPCV2.load_from_checkpoint(weight_path, strict=False)

    arch_instance = utils.get_arch_cells(args.arch_instance)
    model_inn_vae = inn_vae(args, arch_instance).cuda()

    if args.fast_adamax:
        # Fast adamax has the same functionality as torch.optim.Adamax, except it is faster.
        cnn_optimizer = Adamax(model_inn_vae.parameters(),
                               args.learning_rate,
                               weight_decay=args.weight_decay,
                               eps=1e-3)
    else:
        cnn_optimizer = torch.optim.Adamax(model_inn_vae.parameters(),
                                           args.learning_rate,
                                           weight_decay=args.weight_decay,
                                           eps=1e-3)

    cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(cnn_optimizer,
                                                               float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min)

    grad_scalar = GradScaler(2**10)

    num_output = utils.num_output(args.dataset)
    bpd_coeff = 1. / np.log(2.) / num_output

    # if load
    checkpoint_file = os.path.join(args.save, 'checkpoint.pt')
    if args.cont_training:
        logging.info('loading the model.')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        init_epoch = checkpoint['epoch']
        model_inn_vae.load_state_dict(checkpoint['state_dict'])
        model_inn_vae = model_inn_vae.cuda()
        cnn_optimizer.load_state_dict(checkpoint['optimizer'])
        grad_scalar.load_state_dict(checkpoint['grad_scalar'])
        cnn_scheduler.load_state_dict(checkpoint['scheduler'])
        global_step = checkpoint['global_step']
    else:
        global_step, init_epoch = 0, 0

    for epoch in range(init_epoch, args.epochs):
        # update lrs. # todo
        if args.distributed:
            train_dataloader.sampler.set_epoch(global_step + args.seed)
            test_dataloader.sampler.set_epoch(0)

        if epoch > args.warmup_epochs:
            cnn_scheduler.step()

        # Logging.
        logging.info('epoch %d', epoch)

        # Training.
        train_nelbo, global_step = train_inn_vae(train_dataloader, backbone, model_inn_vae,
                                                 cnn_optimizer, grad_scalar, global_step, warmup_iters, writer, logging)
        logging.info('train_nelbo %f', train_nelbo)
        writer.add_scalar('train/nelbo', train_nelbo, global_step)

        model_inn_vae.eval()
        # generate samples less frequently
        eval_freq = 1 if args.epochs <= 50 else 20
        if epoch % eval_freq == 0 or epoch == (args.epochs - 1):
            with torch.no_grad():
                num_samples = 16
                n = int(np.floor(np.sqrt(num_samples)))
                for t in [0.7, 0.8, 0.9, 1.0]:
                    logits = model_inn_vae.sample(num_samples, t)
                    output = model_inn_vae.decoder_output(logits)
                    output_img = output.mean if isinstance(
                        output, torch.distributions.bernoulli.Bernoulli
                    ) else output.sample(t)
                    output_tiled = utils.tile_image(output_img, n)
                    writer.add_image('generated_%0.1f' %
                                     t, output_tiled, global_step)

            valid_neg_log_p, valid_nelbo = test(test_dataloader, backbone,
                                                model_inn_vae, num_samples=10,
                                                args=args, logging=logging)
            logging.info('valid_nelbo %f', valid_nelbo)
            logging.info('valid neg log p %f', valid_neg_log_p)
            logging.info('valid bpd elbo %f', valid_nelbo * bpd_coeff)
            logging.info('valid bpd log p %f', valid_neg_log_p * bpd_coeff)
            writer.add_scalar('val/neg_log_p', valid_neg_log_p, epoch)
            writer.add_scalar('val/nelbo', valid_nelbo, epoch)
            writer.add_scalar('val/bpd_log_p', valid_neg_log_p * bpd_coeff,
                              epoch)
            writer.add_scalar('val/bpd_elbo', valid_nelbo * bpd_coeff, epoch)

        save_freq = int(np.ceil(args.epochs / 100))
        if epoch % save_freq == 0 or epoch == (args.epochs - 1):
            if args.global_rank == 0:
                logging.info('saving the model.')
                torch.save(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model_inn_vae.state_dict(),
                        'optimizer': cnn_optimizer.state_dict(),
                        'global_step': global_step,
                        'args': args,
                        'arch_instance': arch_instance,
                        'scheduler': cnn_scheduler.state_dict(),
                        'grad_scalar': grad_scalar.state_dict()
                    }, checkpoint_file)

    # Final validation
    valid_neg_log_p, valid_nelbo = test(test_dataloader,
                                        model_inn_vae,
                                        num_samples=1000,
                                        args=args,
                                        logging=logging)
    logging.info('final valid nelbo %f', valid_nelbo)
    logging.info('final valid neg log p %f', valid_neg_log_p)
    writer.add_scalar('val/neg_log_p', valid_neg_log_p, epoch + 1)
    writer.add_scalar('val/nelbo', valid_nelbo, epoch + 1)
    writer.add_scalar('val/bpd_log_p', valid_neg_log_p * bpd_coeff, epoch + 1)
    writer.add_scalar('val/bpd_elbo', valid_nelbo * bpd_coeff, epoch + 1)
    writer.close()


def train_inn_vae(args, train_dataloader, backbone, model_inn_vae, cnn_optimizer, grad_scalar,
                  global_step, warmup_iters, writer, logging):
    alpha_i = utils.kl_balancer_coeff(num_scales=model_inn_vae.num_latent_scales,
                                      groups_per_scale=model_inn_vae.groups_per_scale,
                                      fun='square')
    nelbo = utils.AvgrageMeter()
    model_inn_vae.train()
    num_train = len(train_dataloader)

    for step, x, label in enumerate(train_dataloader):
        x = x[0] if len(x) > 1 else x
        x = x.cuda()
        encoded_attributes = label2attribute(label)
        recon, regul_z, regul_r, cross_entropy, mutual_info_wx = \
            model_inn_vae(x, encoded_attributes, num_train)

        alpha = 1
        beta_r = 1
        beta_z = 1
        gamma = 1
        # todo defining nn.parameters for backward propagation
        asso_loss = recon - alpha*mutual_info_wx - \
            beta_z*regul_z - beta_r*regul_r + gamma*cross_entropy

        # todo Loss rebalancing
        # beta_x = 2. / (1 + beta)
        # beta_y = 2. * beta / (1 + beta)
        # loss = beta_x * L_x- beta_y * L_y

        if train_intervention is True:
            intervention_loss = model_inn_vae.intervention()

        # warm-up lr
        if global_step < warmup_iters:
            lr = args.learning_rate * float(global_step) / warmup_iters
            for param_group in cnn_optimizer.param_groups:
                param_group['lr'] = lr

        # sync parameters, it may not be necessary
        if step % 100 == 0:
            utils.average_params(model_inn_vae.parameters(), args.distributed)

        cnn_optimizer.zero_grad()

        with autocast():
            output = model_inn_vae.decoder_output(logits)
            kl_coeff = utils.kl_coeff(
                global_step, args.kl_anneal_portion * args.num_total_iter,
                args.kl_const_portion * args.num_total_iter,
                args.kl_const_coeff)

            recon_loss = utils.reconstruction_loss(output,
                                                   x, crop=model_inn_vae.crop_output)

            balanced_kl, kl_coeffs, kl_vals = utils.kl_balancer(
                kl_all, kl_coeff, kl_balance=True, alpha_i=alpha_i)

            nelbo_batch = recon_loss + balanced_kl
            loss = torch.mean(nelbo_batch)
            norm_loss = model_inn_vae.spectral_norm_parallel()
            bn_loss = model_inn_vae.batchnorm_loss()
            # get spectral regularization coefficient (lambda)
            if args.weight_decay_norm_anneal:
                assert args.weight_decay_norm_init > 0 and args.weight_decay_norm > 0, 'init and final wdn should be positive.'
                wdn_coeff = (1. - kl_coeff) * np.log(
                    args.weight_decay_norm_init) + kl_coeff * np.log(
                        args.weight_decay_norm)
                wdn_coeff = np.exp(wdn_coeff)
            else:
                wdn_coeff = args.weight_decay_norm

            loss += norm_loss * wdn_coeff + bn_loss * wdn_coeff

        grad_scalar.scale(loss).backward()
        utils.average_gradients(model_inn_vae.parameters(), args.distributed)
        grad_scalar.step(cnn_optimizer)
        grad_scalar.update()
        nelbo.update(loss.data, 1)

        if (global_step + 1) % 100 == 0:
            if (global_step + 1) % 1000 == 0:  # reduced frequency
                n = int(np.floor(np.sqrt(x.size(0))))
                x_img = x[:n * n]
                output_img = output.mean if isinstance(
                    output, torch.distributions.bernoulli.Bernoulli
                ) else output.sample()
                output_img = output_img[:n * n]
                x_tiled = utils.tile_image(x_img, n)
                output_tiled = utils.tile_image(output_img, n)
                in_out_tiled = torch.cat((x_tiled, output_tiled), dim=2)
                writer.add_image('reconstruction', in_out_tiled, global_step)

            # norm
            writer.add_scalar('train/norm_loss', norm_loss, global_step)
            writer.add_scalar('train/bn_loss', bn_loss, global_step)
            writer.add_scalar('train/norm_coeff', wdn_coeff, global_step)

            utils.average_tensor(nelbo.avg, args.distributed)
            logging.info('train %d %f', global_step, nelbo.avg)
            writer.add_scalar('train/nelbo_avg', nelbo.avg, global_step)
            writer.add_scalar(
                'train/lr',
                cnn_optimizer.state_dict()['param_groups'][0]['lr'],
                global_step)
            writer.add_scalar('train/nelbo_iter', loss, global_step)
            writer.add_scalar('train/kl_iter', torch.mean(sum(kl_all)),
                              global_step)
            writer.add_scalar(
                'train/recon_iter',
                torch.mean(
                    utils.reconstruction_loss(output, x, crop=model_inn_vae.crop_output)),
                global_step)
            writer.add_scalar('kl_coeff/coeff', kl_coeff, global_step)
            total_active = 0
            for i, kl_diag_i in enumerate(kl_diag):
                utils.average_tensor(kl_diag_i, args.distributed)
                num_active = torch.sum(kl_diag_i > 0.1).detach()
                total_active += num_active

                # kl_ceoff
                writer.add_scalar('kl/active_%d' % i, num_active, global_step)
                writer.add_scalar('kl_coeff/layer_%d' % i, kl_coeffs[i],
                                  global_step)
                writer.add_scalar('kl_vals/layer_%d' % i, kl_vals[i],
                                  global_step)
            writer.add_scalar('kl/total_active', total_active, global_step)

        global_step += 1

    utils.average_tensor(nelbo.avg, args.distributed)
    return nelbo.avg, global_step


def test(val_dataloader, model, num_samples, args, logging):
    if args.distributed:
        dist.barrier()
    nelbo_avg = utils.AvgrageMeter()
    neg_log_p_avg = utils.AvgrageMeter()
    model.eval()
    for step, x in enumerate(val_dataloader):
        x = x[0] if len(x) > 1 else x
        x = x.cuda()

        # change bit length
        x = utils.pre_process(x, args.num_x_bits)

        with torch.no_grad():
            nelbo, log_iw = [], []
            for k in range(num_samples):
                logits, log_q, log_p, kl_all, _ = model(x)
                output = model.decoder_output(logits)
                recon_loss = utils.reconstruction_loss(output,
                                                       x,
                                                       crop=model.crop_output)
                balanced_kl, _, _ = utils.kl_balancer(kl_all, kl_balance=False)
                nelbo_batch = recon_loss + balanced_kl
                nelbo.append(nelbo_batch)
                log_iw.append(
                    utils.log_iw(output,
                                 x,
                                 log_q,
                                 log_p,
                                 crop=model.crop_output))

            nelbo = torch.mean(torch.stack(nelbo, dim=1))
            log_p = torch.mean(
                torch.logsumexp(torch.stack(log_iw, dim=1), dim=1) -
                np.log(num_samples))

        nelbo_avg.update(nelbo.data, x.size(0))
        neg_log_p_avg.update(-log_p.data, x.size(0))

    utils.average_tensor(nelbo_avg.avg, args.distributed)
    utils.average_tensor(neg_log_p_avg.avg, args.distributed)
    if args.distributed:
        # block to sync
        dist.barrier()
    logging.info('val, step: %d, NELBO: %f, neg Log p %f', step, nelbo_avg.avg,
                 neg_log_p_avg.avg)
    return neg_log_p_avg.avg, nelbo_avg.avg
