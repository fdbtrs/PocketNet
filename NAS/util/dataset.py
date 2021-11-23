import os
from sklearn.model_selection import train_test_split

import numbers
import mxnet as mx
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms

from util.config import config as cfg

from pprint import pprint

class MXFaceDataset(Dataset):
    def __init__(self, root_dir, transform):
        super(MXFaceDataset, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __get_classes__(self):
        classes = []
        for idx in range(0, len(self.imgidx)):
            s = self.imgrec.read_idx(idx)
            header, img = mx.recordio.unpack(s)
            label = header.label
            if not isinstance(label, numbers.Number):
                label = label[0]
            classes.append(int(label))
        return classes

    def __len__(self):
        return len(self.imgidx)

def get_train_dataset(root, name):
    """ returns only the train dataset """

    if name == "CASIA":
        train_trans = transforms.Compose ([
            transforms.ToPILImage(),
            transforms.Resize(128), #  128x128
            transforms.RandomCrop(112), # 
            transforms.RandomHorizontalFlip(), # randomly flipping
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        trn_data = MXFaceDataset(root_dir=root, transform=train_trans)
    elif name == "CIFAR-10":
        train_trans = transforms.Compose ([
            transforms.Resize(38), #  128x128
            transforms.RandomCrop(32), # 
            transforms.RandomHorizontalFlip(), # randomly flipping
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        trn_data = CIFAR10(root=root, train=True, download=True, transform=train_trans)
    else:
        trn_data = None

    input_channels = cfg.input_channels
    input_size = cfg.input_size

    n_classes = cfg.n_classes

    return input_size, input_channels, n_classes, trn_data

def get_dataset_without_crop(root, name):
    trans = transforms.Compose ([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if name == "CASIA":
        data = MXFaceDataset(root_dir=root, transform=trans)
    elif name == "CIFAR-10":
        data = CIFAR10(root=root, train=True, download=True, transform=trans)
    else:
        trn_data = None

    return data

# Tested
def get_train_val_split(data, name, val_split=0.5):
    """ returns indexes of a split of the dataset in a stratified manner """
    if name == "CASIA":
        targets = data.__get_classes__()
        targets = targets[1::]
    elif name == "CIFAR-10":
        targets = data.targets

    train_idx, valid_idx = train_test_split(np.arange(len(targets)), test_size=val_split, stratify=targets)

    return train_idx, valid_idx