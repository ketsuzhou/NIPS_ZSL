import os
import yaml
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import torch
import h5py
import os.path as osp
import argparse

def dict2obj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dict2obj(v)
    return d


def arg_completion(args):
    if not os.path.isfile(args.cfg_file):
        raise FileNotFoundError(yamlPath)
    f = open(args.cfg_file, 'r', encoding='utf-8')
    cfg = yaml.load(f.read())

    args = vars(args)
    args = { **args, **cfg}
    args = argparse.Namespace(**args)
    args = dict2obj(args)
    args.FlowBlocks_architecture = eval(args.FlowBlocks_architecture)
    args.use_attn = eval(args.use_attn)
    args.use_split = eval(args.use_split)
    args.downsample = eval(args.downsample)


    if args.dataset == "AwA2":
        args.data_root = "../data/AwA2/"
        args.embading_root = "./data/AwA2/"
        args.attr_dims = 85
        args.nseen = 40
        args.embading_size = [2048, 7, 7]
    elif args.dataset == "CUB":
        args.data_root = "../data/CUB/"
        args.attr_dims = 312
        args.nseen = 150
    elif args.dataset == "SUN":
        args.data_root = "../data/SUN/"
        args.attr_dims = 102
        args.nseen = 645
    else:
        raise NotImplementedError

    args.attribute       = osp.join(args.data_root, "predicate-matrix-continuous.txt")
    args.class_name      = osp.join(args.data_root, "classes.txt")
    args.image           = osp.join(args.data_root, "JPEGImages")
    args.ss_train        = osp.join(args.data_root, "trainclasses.txt")    
    args.ss_test         = osp.join(args.data_root, "testclasses.txt")
    args.ps_train        = osp.join(args.data_root, "proposed_split/trainval_ps.txt")
    args.ps_test_seen    = osp.join(args.data_root, "proposed_split/test_seen_ps.txt")
    args.ps_test_unseen  = osp.join(args.data_root, "proposed_split/test_unseen_ps.txt")
    args.ps_seen_cls     = osp.join(args.data_root, "proposed_split/seen_cls.txt")
    args.ps_unseen_cls   = osp.join(args.data_root, "proposed_split/unseen_cls.txt")

    # postfix = "sa" if self_adaptation else "hybrid"
    # self.test.report_path = osp.join(
    #     self.test.report_base_path, f"{self.ckpt_name}_{self.test.setting}_{self.test.imload_mode}_{postfix}.txt"
    # )
    return args
