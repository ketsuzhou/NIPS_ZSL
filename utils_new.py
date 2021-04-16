import os
import logging
import numpy as np
import torch
from matplotlib import pyplot as plt
import argparse
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
# from generative_classifier import generative_classifier
from data_module import data_module
from data_factory.dataloader import IMAGE_LOADOR
from pl_bolts.models.self_supervised import CPCV2, SSLFineTuner
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import utils
from torch.multiprocessing import Process
import torch.distributed as dist
from VAE import AutoEncoder
from zsl_train_test import train_test
import argparse
import csv
import os
import collections
import pickle
import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from os import path
import logging


logger = logging.getLogger(__name__)


def create_filename(type_, label, args):
    run_label = "_run{}".format(args.run_number) if args.run_number > 0 else ""

    if type_ == "dataset":  # Fixed datasets
        filename = "{}/data/samples/{}".format(args.dir, args.dataset)

    elif type_ == "sample":  # Dynamically sampled from simulator
        filename = "{}/data/samples/{}/{}{}.npy".format(
            args.dir, args.dataset, label, run_label)

    elif type_ == "model":
        filename = "{}/data/models/{}.pt".format(args.dir, args.modelname)

    elif type_ == "checkpoint":
        filename = "{}/data/models/checkpoints/{}_{}_{}.pt".format(
            args.dir, args.modelname, "epoch" if label is None else "epoch_" + label, "{}")

    elif type_ == "resume":
        for label in ["D_", "C_", "B_", "A_", ""]:
            filename = "{}/data/models/checkpoints/{}_epoch_{}last.pt".format(
                args.dir, args.modelname, label, "last")
            if os.path.exists(filename):
                return filename

        raise FileNotFoundError(
            f"Trying to resume training from {filename}, but file does not exist")

    elif type_ == "training_plot":
        filename = "{}/figures/training/{}_{}_{}.pdf".format(
            args.dir, args.modelname, "epoch" if label is None else label, "{}")

    elif type_ == "learning_curve":
        filename = "{}/data/learning_curves/{}.npy".format(
            args.dir, args.modelname)

    elif type_ == "results":
        trueparam_name = "" if args.trueparam is None or args.trueparam == 0 else "_trueparam{}".format(
            args.trueparam)
        filename = "{}/data/results/{}_{}{}.npy".format(
            args.dir, args.modelname, label, trueparam_name)

    elif type_ == "mcmcresults":
        trueparam_name = "" if args.trueparam is None or args.trueparam == 0 else "_trueparam{}".format(
            args.trueparam)
        chain_name = "_chain{}".format(args.chain) if args.chain > 0 else ""
        filename = "{}/data/results/{}_{}{}{}.npy".format(
            args.dir, args.modelname, label, trueparam_name, chain_name)

    elif type_ == "timing":
        filename = "{}/data/timing/{}_{}_{}_{}_{}_{}{}.npy".format(
            args.dir,
            args.algorithm,
            args.outerlayers,
            args.outertransform,
            "mlp" if args.outercouplingmlp else "resnet",
            args.outercouplinglayers,
            args.outercouplinghidden,
            run_label,
        )
    elif type_ == "paramscan":
        filename = "{}/data/paramscan/{}.pickle".format(
            args.dir, args.paramscanstudyname)
    else:
        raise NotImplementedError

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    return filename


def create_modelname(args, model_name):
    run_label = "_run{}".format(args.run_number) if args.run_number > 0 else ""

    args.modelname = "{}_{}_{}".format(
        model_name, args.dataset, run_label)


def nat_to_bit_per_dim(dim):
    if isinstance(dim, (tuple, list, np.ndarray)):
        dim = np.product(dim)
    logger.debug("Nat to bit per dim: factor %s", 1.0 / (np.log(2) * dim))
    return 1.0 / (np.log(2) * dim)


def sum_except_batch(x, num_batch_dims=1):
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


def array_to_image_folder(data, folder):
    for i, x in enumerate(data):
        x = np.clip(np.transpose(x, [1, 2, 0]) / 256.0, 0.0, 1.0)
        if i == 0:
            logger.debug("x: %s", x)
        plt.imsave(f"{folder}/{i}.jpg", x)


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
    config.in_shape = [2048, 7, 7]

    if not os.path.exists(config.result_root):
        os.makedirs(config.result_root)
    # if not os.path.exists(config.model_root):
    # os.makedirs(config.model_root)
    # namespace ==> dictionary
    return config


def fix_act_norm_issue(model):
    if isinstance(model, ActNorm):
        logger.debug("Fixing initialization state of actnorm layer")
        model.initialized = True

    for _, submodel in model._modules.items():
        fix_act_norm_issue(submodel)


def make_training_kwargs(args):
    dm = data_module(args)
    if args.zsl_type == "conventional":
        train_loader, val_loader = dm.conventional_dataloader()
    elif args.zsl_type == "generalized":
        train_loader, val_loader = dm.generalized_dataloader()
    else:
        NotImplementedError

    kwargs = {
        "batch_size": args.batch_size,
        "initial_lr": args.initial_learning_rate,
        "scheduler": optim.lr_scheduler.CosineAnnealingLR,
        "clip_gradient": args.clip,
        "seed": args.seed + args.run_number,
        "train_loader": train_loader,
        "val_loader": val_loader
    }
    if args.weight_decay is not None:
        kwargs["optimizer_kwargs"] = {"weight_decay": float(args.weight_decay)}

    return kwargs


def read_attribute():
    class_path = './AWA2_attribute.pkl'
    with open(class_path, 'rb') as f:
        w2v = pickle.load(f)

    w2v = torch.tensor(w2v).float()
    U, s, V = torch.svd(w2v)
    # reconstruct = torch.mm(torch.mm(U,torch.diag(s)),torch.transpose(V,1,0))

    w2v_att = torch.transpose(V, 1, 0)
    att = torch.mm(U, torch.diag(s))
    normalize_att = torch.mm(U, torch.diag(s))
    dim_v = V.size(1)

    return dim_v, w2v_att, att, normalize_att
