import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import torch
import h5py
from torchvision import transforms as transform_lib
from data_factory.proposed_split import PSFactory
from data_factory.standard_split import SSFactory
from torch.utils.data import DataLoader
import torch
from pytorch_lightning import LightningDataModule
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization

class IMAGE_LOADOR(LightningDataModule): 
    def __init__(self, image_dir, num_workers, attribute, *args, **kwargs,):
        super().__init__(*args, **kwargs)
        self.image_dir = image_dir
        self.num_workers = num_workers
        self.attribute = attribute

    def train_dataloader(self, split, batch_size):
        transforms = self._default_transforms() if self.train_transforms is None \
             else self.train_transforms

        if split == "SS":
            dataset = SSFactory(
                self.image_dir, self.attribute, self.class_name, args.ss_train, 
                transform=transforms, batch_size=self.batch_size, im_size=self.im_size
            )
        elif args.split == "PS":
            dataset = PSFactory(
                args.image_dir, args.attribute, args.class_name, args.ps_train, 
                transform=transforms, batch_size=self.batch_size, im_size=self.im_size
            )
        else:
            raise NotImplementedError

        dataset.im_size = self.im_size
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return data_loader

    def val_dataloader(self, batch_size, num_images_per_class=50, add_normalize=False):
        transforms = self._default_transforms() if self.val_transforms is None else self.val_transforms

        dataset = UnlabeledImagenet(self.data_dir,
                                    num_imgs_per_class_val_split=num_images_per_class,
                                    meta_dir=self.meta_dir,
                                    split='val',
                                    transform=transforms)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    def test_dataloader(self, batch_size, num_images_per_class, add_normalize=False):
        transforms = self._default_transforms() if self.test_transforms is None else self.test_transforms

        dataset = UnlabeledImagenet(self.data_dir,
                                    num_imgs_per_class=num_images_per_class,
                                    meta_dir=self.meta_dir,
                                    split='test',
                                    transform=transforms)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def _default_transforms(self):
        mnist_transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            imagenet_normalization()
        ])
        return mnist_transforms


from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
def data_transform(name, size):
    name = name.strip().split('+')
    name = [n.strip() for n in name]
    transform = []
    
    # Loading Method:
    if "resize_random_crop" in name:
        transform.extend([
            transforms.Resize(int(size * 8. / 7.)), # 224 -> 256
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(0.5)
            ])
    elif "resize" in name:
        transform.extend([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(0.5)
            ])
    else:
        # "auto"
        transform.extend([
            transforms.Resize(size),
            transforms.CenterCrop(size)
            ])
    
    if "colorjitter" in name:
        transform.append(
            transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.2))

    transform.extend([transforms.ToTensor(), normalize])
    transform = transforms.Compose(transform)
    return transform
    



def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label


class EMBADING_LOADER(object):
    def __init__(self, opt):

        if opt.dataset == 'ImageNet':
            self.read_matimagenet(opt)
        else:
            self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.feature_dim = self.train_feature.shape[1]
        self.att_dim = self.attribute.shape[1]
        self.text_dim = self.att_dim
        self.tr_cls_centroid = np.zeros([self.seenclasses.shape[0], self.feature_dim], np.float32)
        for i in range(self.seenclasses.shape[0]):
            self.tr_cls_centroid[i] = np.mean(self.train_feature[self.train_label == i].numpy(), axis=0)


    def read_matimagenet(self, opt):
        if opt.preprocessing:

            print('MinMaxScaler...')

            scaler = preprocessing.MinMaxScaler()
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = scaler.fit_transform(np.array(matcontent['features']))
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = scaler.transform(np.array(matcontent['features_val']))
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()
            print('load train feature')
            matcontent = h5py.File('/media/guyuchao/data/gyc/PycharmProject/Zero-shot-cls/VAE_cFlow/data/ImageNet/lp500/lp500.mat', 'r')
            feature_unseen = scaler.transform(np.array(matcontent['features']))
            label_unseen = np.array(matcontent['labels']).astype(int).squeeze() - 1
            matcontent.close()
            print('load test feature')
        else:
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = np.array(matcontent['features'])
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = np.array(matcontent['features_val'])
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()

        matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
        self.attribute = np.array(matcontent['w2v']).T
        l2normalizer = preprocessing.Normalizer(norm='l2')
        self.attribute = l2normalizer.fit_transform(self.attribute)

        self.train_feature = torch.from_numpy(feature).float()

        self.train_label = torch.from_numpy(label).long()

        self.test_seen_feature = torch.from_numpy(feature_val).float()

        self.test_seen_label = torch.from_numpy(label_val).long()
        self.test_unseen_feature = torch.from_numpy(feature_unseen).float()

        self.test_unseen_label = torch.from_numpy(label_unseen).long()

        self.ntrain = self.train_feature.size()[0]
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))

        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.train_class = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_label = map_label(self.train_label, self.seenclasses)
        self.test_unseen_label = map_label(self.test_unseen_label, self.unseenclasses)
        self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)
        self.train_att = self.attribute[self.seenclasses]
        self.test_att = self.attribute[self.unseenclasses]


    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        #train_loc = matcontent['train_loc'].squeeze() - 1
        #val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        if not opt.validation:
            scaler = preprocessing.MinMaxScaler()
            _train_feature = scaler.fit_transform(feature[trainval_loc])
            _test_seen_feature = scaler.transform(feature[test_seen_loc])
            _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
            #_train_feature = feature[trainval_loc]
            #_test_seen_feature = feature[test_seen_loc]
            #_test_unseen_feature = feature[test_unseen_loc]
            self.train_feature = torch.from_numpy(_train_feature).float()

            mx = self.train_feature.max()

            self.train_feature.mul_(1 / mx)
            self.train_label = torch.from_numpy(label[trainval_loc]).long()
            self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
            self.test_unseen_feature.mul_(1 / mx)
            self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
            self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()

            #dsa
            self.test_seen_feature.mul_(1 / mx)
            self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

        else:
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))

        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.train_label = map_label(self.train_label, self.seenclasses)
        self.test_unseen_label = map_label(self.test_unseen_label, self.unseenclasses)
        self.test_seen_label = map_label(self.test_seen_label, self.seenclasses)
        self.train_att = self.attribute[self.seenclasses].numpy()
        self.test_att  = self.attribute[self.unseenclasses].numpy()



