from pytorch_lightning import LightningDataModule
import os
import numpy as np
from torch.utils import data
import torch
from PIL import Image
from torch.utils import data
from data_factory.data_transform import TrainTransforms, TestTransforms


def image_load(class_file, label_file):
    with open(class_file, 'r') as f:
        class_names = [l.strip() for l in f.readlines()]
    class_map = {}
    for i, l in enumerate(class_names):
        items = l.split()
        class_map[items[-1]] = i
    #print(class_map)
    all_data_path = []
    labels = {}
    with open(label_file, 'r') as f:
        image_label = [l.strip() for l in f.readlines()]
    for lines in image_label:
        items = lines.split()
        all_data_path.append(items[0])
        labels[items[0]] = int(items[1])
    return all_data_path, labels, class_map


def split_byclass(config, all_data_path, labels, attributes, class_map):
    with open(config['train_classes'], 'r') as f:
        train_lines = [l.strip() for l in f.readlines()]
    with open(config['test_classes'], 'r') as f:
        test_lines = [l.strip() for l in f.readlines()]
    train_attr = []
    test_attr = []
    train_class_set = {}
    for i, name in enumerate(train_lines):
        idx = class_map[name]
        train_class_set[idx] = i
        # idx is its real label
        train_attr.append(attributes[idx])
    test_class_set = {}
    for i, name in enumerate(test_lines):
        idx = class_map[name]
        test_class_set[idx] = i
        test_attr.append(attributes[idx])
    train = []
    test = []
    label_map = {}
    for ins in all_data_path:
        v = labels[ins]
        # inital label
        if v in train_class_set:
            train.append(ins)
            label_map[ins] = train_class_set[v]
        else:
            test.append(ins)
            label_map[ins] = test_class_set[v]
    train_attr = torch.from_numpy(np.array(train_attr, dtype='float')).float()
    test_attr = torch.from_numpy(np.array(test_attr, dtype='float')).float()
    return train, test, label_map, train_attr, test_attr


class DataSet(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, image_dir, num_classes, data_path, labels, transform,
                 is_train):
        'Initialization'
        self.labels = labels
        self.data_path = data_path
        self.transform = transform
        self.image_dir = image_dir
        self.num_classes = num_classes
        self.is_train = is_train

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_path)

    def __getitem__(self, idx):
        'Generates one sample of data'
        id = self.data_path[idx]
        # Convert to RGB to avoid png.
        X = Image.open(self.image_dir + id).convert('RGB')
        X = self.transform(X)
        label = self.labels[id]
        return X, label


class data_model(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.train_transforms = TrainTransforms
        self.test_transforms = TestTransforms
        self.all_data_path, self.labels, self.class_map = image_load(
            args.class_file, args.image_label)
        self.train_data_path, self.test_data_path, self.label_map, self.train_attr, self.test_attr = split_byclass(
            args, self.all_data_path, self.labels,
            np.loadtxt(args.attributes_file), self.class_map)
            
        self.num_classes = self.train_attr.size(0)

    def train_dataloader(self):
        params = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'shuffle': True,
            'sampler': None
        }
        train_dataloader = data.DataLoader(
            DataSet(self.image_dir,
                    self.num_classes,
                    self.train_data_path,
                    self.labels,
                    self.train_transforms,
                    is_train=True), **params)

        return train_dataloader

    def val_dataloader(self):
        params = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'sampler': None,
            'shuffle': False
        }
        val_dataloader = data.DataLoader(
            DataSet(self.image_dir,
                    self.num_classes,
                    self.test__data_path,
                    self.labels,
                    self.test_transforms,
                    is_train=False), **params)

        return val_dataloader

    def test_dataloader(self):
        params = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'sampler': None,
            'shuffle': False
        }
        test_dataloader = data.DataLoader(
            DataSet(self.image_dir,
                    self.num_classes,
                    self.test__data_path,
                    self.labels,
                    self.test_transforms,
                    is_train=False), **params)

        return test_dataloader