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


class discriminative_classifier(nn.Module):
    def __init__(self, dim_v, w2v_att, att, normalize_att, trainable_w2v=False,
                 non_linear_act=False, normalize_V=False, normalize_F=False):
        super(discriminative_classifier, self).__init__()
        # weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpcv2_weights/checkpoints/epoch%3D526.ckpt'
        weight_path = '../../.cache/torch/hub/checkpoints/epoch%3D526.ckpt'
        self.backbone = CPCV2.load_from_checkpoint(weight_path, strict=False)

        self.dim_f = 2048
        self.dim_v = dim_v
        self.dim_att = att.shape[1]
        self.nclass = att.shape[0]
        self.hidden = self.dim_att//2
        self.w2v_att = w2v_att
        self.non_linear_act = non_linear_act
        self.normalize_att = normalize_att

        self.W_1 = nn.Parameter(nn.init.normal_(
            torch.empty(self.dim_v, self.dim_f)), requires_grad=True)
        self.W_2 = nn.Parameter(nn.init.zeros_(
            torch.empty(self.dim_v, self.dim_f)), requires_grad=True)
        self.W_3 = nn.Parameter(nn.init.zeros_(
            torch.empty(self.dim_v, self.dim_f)), requires_grad=True)

        self.normalize_V = normalize_V
        self.normalize_F = normalize_F

        self.w2v_att = F.normalize(torch.tensor(w2v_att))
        self.V = nn.Parameter(self.w2v_att.clone(),
                              requires_grad=trainable_w2v)

        self.att = nn.Parameter(F.normalize(
            torch.tensor(att)), requires_grad=False)

        # Compute the similarity between classes
        self.P = torch.mm(self.att, torch.transpose(self.att, 1, 0))
        assert self.P.size(1) == self.P.size(
            0) and self.P.size(0) == self.nclass

    def feature_extraction(self, inputs):
        with torch.no_grad():
            feature_map = self.backbone(inputs)

        return feature_map

    def forward(self, inputs, labels):
        feature_map = self.feature_extraction(inputs)

        shape = feature_map.shape
        Fm = feature_map.reshape(shape[0], shape[1], -1)

        if self.normalize_V:
            V = F.normalize(self.V)
        else:
            V = self.V

        if self.normalize_F:
            Fm = F.normalize(Fm, dim=1)

        # Compute attributes score on each image patch
        patchwise_attributes_score = torch.einsum(
            'av, vf, bfp -> bap', V, self.W_1, Fm)

        # sigmoid encourages sparsity for patchwise_attributes_score
        if self.is_sigmoid:
            patchwise_attributes_score = torch.sigmoid(
                patchwise_attributes_score)

        # compute attention map over patches
        attention_over_patches = torch.einsum(
            'av, vf, bfp -> bap', V, self.W_2, Fm)
        # performing softmax at meanwhile causes there is no attention over attributes
        attention_over_patches = F.softmax(attention_over_patches, dim=-1)

        # compute attribute-attented feature map
        # todo
        Fm_attented_over_patches = torch.einsum(
            'bap, bfp -> baf', attention_over_patches, Fm)

        # compute attention over attributes
        attention_over_attributes = torch.einsum(
            'av, vf, baf -> ba', V, self.W_3, Fm_attented_over_patches)
        # sigmoid encourages sparsity for attention_over_attributes
        attention_over_attributes = torch.sigmoid(attention_over_attributes)

        # compute attribute scores from attribute attention maps
        As_attented_over_patches = torch.einsum(
            'bap, bap -> ba', attention_over_patches, patchwise_attributes_score)

        if self.non_linear_act:
            As_attented_over_patches = F.relu(As_attented_over_patches)

        # compute the final prediction as the product of semantic scores, attribute scores, and attention over attribute scores
        # self.att is the strength of having the attribute a in class c,
        Wv_attented_over_attributes = torch.einsum(
            'la, ba, ba -> bal', self.att, attention_over_attributes, As_attented_over_patches)

        # [bl] <== [bal]
        class_score = torch.sum(
            Wv_attented_over_attributes, axis=1)

        # cross entropy loss
        prob = nn.LogSoftmax(class_score, dim=1)
        loss = - torch.einsum('bl, bl -> b', prob, labels)
        loss_CE = torch.mean(loss)

        #
        predicted_attributes = torch.einsum(
            'av, vf, baf -> ba', V, self.W_1, Fm_attented_over_patches)

        return loss_CE, attention_over_patches, attention_over_attributes, patchwise_attributes_score

    def intervention_on_patches(self, labels, attention_over_attributes, attention_over_patches, patchwise_attributes_score):
        ################################################################
        # todo add constain about the mutual exclusion of spacial attention of different attributes

        # predicted_class = prob.argmax(dim=1) # [b] <== [bc]
        # predicted_class_attributes = self.att[predicted_class] #[ba]
        # predicted_attributes = torch.einsum(
        #     'av, vf, baf -> ba', V, self.W_1, Fm_attented_over_patches)
        # attention_over_patches1 = torch.einsum(
        #     'ba, ba, bar -> bar', predicted_attributes, attention_over_attributes, patchwise_attributes_score)

        with torch.no_grad():
            #############################################
            # following gets reliable patch_marsk responsible for the change of attributes

            k = torch.einsum('bl -> b', labels)
            for i in range(attention_over_attributes.size(0)):
                a, idx = torch.topk(attention_over_attributes[i, :], k[i])
                b = [i.tolist() for i in labels[i].nonzero()]
                b = set([n for a in b for n in a])
                idx_intervention_attributes = set(idx.tolist()).intersection(b)

            # compute attribute scores from attribute attention maps
            # w2 w1
            patchwise_attributes_score = torch.einsum(
                'bap, bap -> bap', attention_over_patches, patchwise_attributes_score)

            patch_marsk = self.get_patch_marsk(patchwise_attributes_score)
            # reverse_patch_marsk = 1 - patch_marsk

############################################
# following calculates the statistics afeter intervention
            As_marsked_over_patches = torch.einsum(
                'bap, bap -> ba ', patch_marsk, patchwise_attributes_score)

            if self.non_linear_act:
                As_marsked_over_patches = F.relu(As_marsked_over_patches)

            # todo exploiting the connection to the knowledge distillation, Softmax T
            Fm_attented_over_patches = torch.einsum(
                'bap, bfp -> baf', patch_marsk, Fm)

            # compute attention over attributes, the attention of intervented attribute should be small, v has dim 300
            intermediate = torch.einsum(
                'av, vf, baf -> bav', self.V, self.W_3, Fm_attented_over_patches)

            changed_attention_over_attributes = torch.sigmoid(
                torch.einsum('bav-> ba', intermediate))
############################################

            # Wv_attented_over_attributes = torch.einsum(
            #     'sa, ba, ba -> bas', self.att, changed_attention_over_attributes, As_attented_over_patches)

            # class_score = torch.einsum('bas -> bs', Wv_attented_over_attributes)

            # # cross entropy loss
            # # 1 in each line of labels indicates the present of an attribute
            # prob = nn.LogSoftmax(class_score, dim=1)
            # loss = - torch.einsum('bs, bs -> b', prob, labels)
            # loss_CE = torch.mean(loss)
###########################################

            predicted_cluster_center = torch.einsum('bav -> bv', intermediate)

            Wv_attented_over_attributes = torch.einsum(
                'sa, ba, ba -> bas', self.att, changed_attention_over_attributes, As_attented_over_patches)

            # inn to bv

            return idx_intervention_attributes, patch_marsk


def get_patch_marsk(self, patchwise_attributes_score, ):
    # todo binarizing patch_marsk by uing threshold or by determining a patch number for each attribute
    patch_marsk = torch.einsum('bap ', patchwise_attributes_score)
