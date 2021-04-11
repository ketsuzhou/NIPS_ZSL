import torchvision.transforms as transforms
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from torchvision import transforms as transform_lib
import pickle
import scipy.io as sio
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset, Subset
from pytorch_lightning import LightningDataModule
import os
import numpy as np
from torch.utils import data
import torch
import torchvision
from PIL import Image
from torch.utils import data
from data_factory.data_transform import TrainTransforms, TestTransforms
torchvision.datasets.ImageNet


def normalization(dataMat):
    meanVal = torch.mean(dataMat, axis=0)
    newData = dataMat-meanVal
    m = torch.max(torch.einsum('ij, ij -> i ', newData, newData))
    newData /= m
    return newData


def PCA_svd(X, k, center=True):
    X = X.cuda()
    n = X.size()[0]
    ones = torch.ones(n).view([n, 1])
    h = ((1/n) * torch.mm(ones, ones.t())
         ) if center else torch.zeros(n*n).view([n, n])
    H = torch.eye(n) - h
    H = H.cuda()
    X_center = torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components = v[:k].t()
    # explained_variance = torch.mul(s[:k], s[:k])/(n-1)
    return components


def image_load(class_file, label_file):
    with open(class_file, 'r') as f:
        class_names = [l.strip() for l in f.readlines()]
    class_map = {}
    for i, l in enumerate(class_names):
        items = l.split()
        class_map[items[-1]] = i
    # print(class_map)
    all_data_path = []
    labels = {}
    with open(label_file, 'r') as f:
        image_label = [l.strip() for l in f.readlines()]
    for lines in image_label:
        items = lines.split()
        all_data_path.append(items[0])
        labels[items[0]] = int(items[1])
    return all_data_path, labels, class_map


def split_byclass(seen_classes, unseen_classes, all_data_path, labels,
                  class_map):
    with open(seen_classes, 'r') as f:
        seen_lines = [l.strip() for l in f.readlines()]

    with open(unseen_classes, 'r') as f:
        unseen_lines = [l.strip() for l in f.readlines()]

    seen_class_set = {}

    # test_attr.append(attributes[idx])
    for i, name in enumerate(seen_lines):
        idx = class_map[name]
        seen_class_set[idx] = i
        # idx is its real label

    num_seen_class_set = {}
    for i, name in enumerate(unseen_lines):
        idx = class_map[name]
        num_seen_class_set[idx] = i

    seen = []
    unseen = []
    label_map = {}
    for ins in all_data_path:
        v = labels[ins]
        # inital label
        if v in seen_class_set:
            seen.append(ins)
            label_map[ins] = seen_class_set[v]
        else:
            unseen.append(ins)
            label_map[ins] = num_seen_class_set[v]

    return seen, unseen


class DataSet(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, image_dir, data_path, labels, transform):
        'Initialization'
        self.labels = labels
        self.data_path = data_path
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_path)

    def __getitem__(self, idx):
        'Generates one sample of '
        data_path_name = self.data_path[idx]
        # # Convert to RGB to avoid png.
        img = Image.open(self.image_dir +
                         data_path_name).convert('RGB')
        img = self.transform(img)
        label = self.labels[data_path_name]
        return img, label


def read_attribute(attributes_file):
    attributes = []
    with open(attributes_file, 'r') as f:
        for line in f.readlines():
            # attributes = torch.cat(attributes, torch.tensor(
            #     [int(x) for x in line.split()]), dim=0)
            attributes.append(
                [float(x) for x in line.split()])
    return attributes


class data_module():
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.train_transforms = TrainTransforms()
        self.test_transforms = TestTransforms()

        self.num_workers = args.num_workers
        self.image_dir = os.path.join(
            args.data_root, args.dataset, 'JPEGImages/')
        class_file = os.path.join(args.data_root, args.dataset, 'classes.txt')
        image_label = os.path.join(args.data_root, args.dataset,
                                   'image_labels.txt')
        binary_attributes_file = os.path.join(args.data_root, args.dataset,
                                              'predicate-matrix-binary.txt')
        continuous_attributes_file = os.path.join(args.data_root, args.dataset,
                                                  'predicate-matrix-continuous.txt')
        self.continuous_attributes = read_attribute(continuous_attributes_file)
        self.binary_attributes = read_attribute(binary_attributes_file)
        self.num_classes = len(self.binary_attributes)
        self.binary_attributes = torch.tensor(self.binary_attributes)

        # matcontent = sio.loadmat(
        #     args.data_root + "/" + args.dataset + "/" + "att_splits.mat")
        # self.attribute = torch.from_numpy(matcontent['att'].T).float()

        class_path = './AWA2_attribute.pkl'
        with open(class_path, 'rb') as f:
            w2v = pickle.load(f)

        w2v = torch.tensor(w2v)
        self.normalized_w2v = normalization(w2v).to(torch.float32)

        class_embedings = torch.einsum(
            'ca, ad -> cd', self.binary_attributes, self.normalized_w2v)

        intervented_embedings = {}
        for i in range(len(self.binary_attributes)):
            index = (self.binary_attributes[i] == 1).nonzero().squeeze(1)
            intervented_embedings[f'intervention_for_class_{i}'] = class_embedings[i] - \
                self.normalized_w2v[index]
        self.intervented_embedings = intervented_embedings
        
        self.all_data_path, self.all_data_labels, self.name2label = image_load(
            class_file, image_label)

        seen_classes = os.path.join(args.data_root, args.split, args.dataset,
                                    'trainvalclasses.txt')
        unseen_classes = os.path.join(args.data_root, args.split, args.dataset,
                                      'testclasses.txt')
        self.seen_data_path, self.unseen_data_path = split_byclass(
            seen_classes, unseen_classes, self.all_data_path,
            self.all_data_labels, self.name2label)

        self.train_params = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'shuffle': True,
            'sampler': None
        }
        self.test_params = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'sampler': None,
            'shuffle': False
        }

    def conventional_dataloader(self):
        train_dataloader = data.DataLoader(
            DataSet(self.image_dir,
                    self.seen_data_path,
                    self.all_data_labels,
                    self.train_transforms,
                    ), **self.train_params)

        test_dataloader = data.DataLoader(
            DataSet(self.image_dir,
                    self.unseen_data_path,
                    self.all_data_labels,
                    self.train_transforms
                    ), **self.test_params)

        return train_dataloader, test_dataloader

    def random_split(self, dataset, test_fraction=0.5, split_seed=0):
        train_idx, test_idx = train_test_split(range(len(dataset)),
                                               test_size=test_fraction,
                                               random_state=split_seed)
        train = Subset(dataset, train_idx)
        test = Subset(dataset, test_idx)
        return train, test

    def generalized_train_dataloader(self):
        dataset = DataSet(self.image_dir,
                          self.seen_data_path,
                          self.all_data_labels,
                          self.train_transforms)
        seen_train_dataset, seen_test_dataset = self.random_split(
            dataset, test_fraction=0.5, split_seed=0)
        train_dataloader = data.DataLoader(seen_train_dataset,
                                           **self.train_params)

        unseen_test_dataset = DataSet(self.image_dir,
                                      self.unseen_data_path,
                                      self.all_data_labels,
                                      self.test_transforms)
        test_dataset = ConcatDataset([seen_test_dataset, unseen_test_dataset])
        test_dataloader = data.DataLoader(test_dataset, **self.test_params)
        return train_dataloader, test_dataloader
