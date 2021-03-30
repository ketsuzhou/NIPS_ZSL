"""Train Flow++ on CIFAR-10.

Train script adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
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
from Generative_Classifier import Generative_Classifier
from Data_Modul import Data_Module
from data_factory.dataloader import IMAGE_LOADOR
from pl_bolts.models.self_supervised import CPCV2, SSLFineTuner
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import utils
from torch.multiprocessing import Process
import torch.distributed as dist
from VAE import AutoEncoder


def arg_parse():
    parser = argparse.ArgumentParser(description='Flow++ on CIFAR-10')

    parser.add_argument(
        '--cfg_file',
        type=str,
        dest='cfg_file',
        default='configs/self_adaptation/ResNet101_AwA2_SS_C.yaml')

    # data loader params
    parser.add_argument('--data_root',
                        default='../data',
                        type=str,
                        metavar='DIRECTORY',
                        help='path to data directory')
    parser.add_argument('--dataset',
                        type=str,
                        default='AwA2',
                        choices=[],
                        help='which dataset to use')
    parser.add_argument('--split',
                        default='standard_split',
                        metavar='NAME',
                        help='image dir for loading images')
    parser.add_argument('--validation',
                        action='store_true',
                        default=False,
                        help='enable cross validation mode')

    # experimental results
    parser.add_argument('--result_root',
                        type=str,
                        default='./results',
                        help='location of the results')
    parser.add_argument('--save',
                        type=str,
                        default='exp',
                        help='id used for storing intermediate results')
    parser.add_argument('--use_pre',
                        action='store_true',
                        default=False,
                        help='use pre-extracted feature')

    # Flow params
    parser.add_argument('--num_channels',
                        default=96,
                        type=int,
                        help='Number of channels in Flow++')
    parser.add_argument('--num_ConvAttnBlock',
                        default=1,
                        type=int,
                        help='Number of ConvAttnBlock')
    parser.add_argument('--num_components',
                        default=2,
                        type=int,
                        help='Number of components in the mixture')
    parser.add_argument('--use_self_attn',
                        type=bool,
                        default=True,
                        help='Use attention in the coupling layers')

    # optimization
    parser.add_argument('--batch_size',
                        type=int,
                        default=2,
                        help='batch size per GPU')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-2,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min',
                        type=float,
                        default=1e-4,
                        help='min learning rate')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=3e-4,
                        help='weight decay')
    parser.add_argument(
        '--weight_decay_norm',
        type=float,
        default=0.,
        help='The lambda parameter for spectral regularization.')
    parser.add_argument('--weight_decay_norm_init',
                        type=float,
                        default=10.,
                        help='The initial lambda parameter')
    parser.add_argument(
        '--weight_decay_norm_anneal',
        action='store_true',
        default=False,
        help='This flag enables annealing the lambda coefficient from '
        '--weight_decay_norm_init to --weight_decay_norm.')
    parser.add_argument('--epochs',
                        type=int,
                        default=200,
                        help='num of training epochs')
    parser.add_argument('--warmup_epochs',
                        type=int,
                        default=5,
                        help='num of training epochs in which lr is warmed up')
    parser.add_argument('--fast_adamax',
                        action='store_true',
                        default=False,
                        help='This flag enables using our optimized adamax.')
    parser.add_argument('--arch_instance',
                        type=str,
                        default='res_mbconv',
                        help='path to the architecture instance')
    parser.add_argument('--drop_prob',
                        type=float,
                        default=0.2,
                        help='Dropout probability')

    parser.add_argument('--num_samples',
                        default=64,
                        type=int,
                        help='Number of samples at test time')
    parser.add_argument('--num_workers',
                        default=4,
                        type=int,
                        help='Number of data loader threads')
    parser.add_argument('--resume',
                        type=bool,
                        default=False,
                        help='Resume from checkpoint')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--save_dir',
                        type=str,
                        default='samples',
                        help='Directory for saving samples')

    parser.add_argument('--class_embedding', default='att', type=str)

    # DDP.
    parser.add_argument('--node_rank',
                        type=int,
                        default=0,
                        help='The index of node.')
    parser.add_argument('--num_proc_node',
                        type=int,
                        default=1,
                        help='The number of nodes in multi node env.每台机器进程数')
    parser.add_argument('--local_rank',
                        type=int,
                        default=1,
                        help='rank of process in the node, 每台机子上使用的GPU的序号')
    parser.add_argument('--global_rank',
                        type=int,
                        default=0,
                        help='rank of process among all the processes')
    parser.add_argument('--num_process_per_node',
                        type=int,
                        default=1,
                        help='number of gpus')
    parser.add_argument(
        '--master_address',
        type=str,
        default='127.0.0.1',
        help=
        'address for master, master节点相当于参数服务器，其会向其他卡广播其参数,rank=0的进程就是master进程')
    parser.add_argument('--seed',
                        type=int,
                        default=2,
                        help='seed used for initialization')

    args = parser.parse_args()
    args.save = args.result_root + '/eval-' + args.save
    utils.create_exp_dir(args.save)
    args = config_process(args)
    return args


def config_process(config):
    if not os.path.isfile(config.cfg_file):
        raise FileNotFoundError()
    f = open(config.cfg_file, 'r', encoding='utf-8')
    cfg = yaml.load(f.read())

    config = vars(config)
    config = {**config, **cfg}
    config = argparse.Namespace(**config)
    config = dict2obj(config)
    config.FlowBlocks_architecture = eval(config.FlowBlocks_architecture)
    config.use_attn = eval(config.use_attn)
    config.use_split = eval(config.use_split)
    config.downsample = eval(config.downsample)

    if config.dataset == "AwA2":
        config.attr_dims = 85
        config.nseen = 40
    elif config.dataset == "CUB":
        config.attr_dims = 312
        config.nseen = 150
    elif config.dataset == "SUN":
        config.attr_dims = 102
        config.nseen = 645
    else:
        raise NotImplementedError

    config.image_dir = os.path.join(config.data_root, config.dataset,
                                    'images/')
    config.class_file = os.path.join(config.data_root, config.dataset,
                                     'classes.txt')
    config.image_label = os.path.join(config.data_root, config.dataset,
                                      'image_labels.txt')
    config.attributes_file = os.path.join(config.data_root, config.dataset,
                                          'class_attributes.txt')

    config.train_classes = os.path.join(config.data_root, config.split,
                                        config.dataset, 'trainvalclasses.txt')
    config.test_classes = os.path.join(config.data_root, config.split,
                                       config.dataset, 'testclasses.txt')

    if not os.path.exists(config.result_root):
        os.makedirs(config.result_root)
    if not os.path.exists(config.model_root):
        os.makedirs(config.model_root)
    #namespace ==> dictionary
    return config


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6020'
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            rank=rank,
                            world_size=size)
    fn(args)
    dist.destroy_process_group()


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logging = utils.Logger(args.global_rank, args.save)
    writer = utils.Writer(args.global_rank, args.save)

    arch_instance = utils.get_arch_cells(args.arch_instance)
    model = AutoEncoder(args, writer, arch_instance)

    # Get data loaders.
    dm = Data_Module(args)
    train_dataloader = dm.train_dataloader
    val_dataloader = dm.val_dataloader
    test_dataloader = dm.test_dataloader

    args.num_total_iter = len(train_dataloader) * args.epochs
    warmup_iters = len(train_dataloader) * args.warmup_epochs
    swa_start = len(train_dataloader) * (args.epochs - 1)

    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpcv2_weights/checkpoints/epoch%3D526.ckpt'
    cpc_v2 = CPCV2.load_from_checkpoint(weight_path, strict=False)
    cpc_v2.freeze()
    # finetuner
    finetuner = SSLFineTuner(cpc_v2,
                             in_features=cpc_v2.z_dim,
                             num_classes=cpc_v2.num_classes)
    from pl_bolts.models.autoencoders import VAE

    # train
    trainer = pl.Trainer(num_processes=2)
    trainer.fit(finetuner, dm)

    # test
    trainer.test(datamodule=dm)

    generative_classifier = Generative_Classifier(args)


if __name__ == '__main__':
    # Set seeds
    args = arg_parse()
    size = args.num_process_per_node
    if size > 1:
        args.distributed = True
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' %
                  (args.node_rank, rank, global_rank))
            p = Process(target=init_processes,
                        args=(global_rank, global_size, main, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # for debugging
        print('starting in debug mode')
        args.distributed = True
        init_processes(0, size, main, args)
