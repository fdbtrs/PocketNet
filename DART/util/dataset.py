import os
from sklearn.model_selection import train_test_split

import numbers
import mxnet as mx
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

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

    def __len__(self):
        return len(self.imgidx)


def get_train_dataset(root):
    """ returns only the train dataset """

    train_trans = transforms.Compose ([
        transforms.ToPILImage(),
        transforms.Resize(128), #  128x128
        transforms.RandomCrop(112), # 
        transforms.RandomHorizontalFlip(), # randomly flipping
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


    trn_data = MXFaceDataset(root_dir=root, transform=train_trans)

    return trn_data

def get_casia_without_crop(root):
    trans = transforms.Compose ([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    data = MXFaceDataset(root_dir=root, transform=trans)

    return data

# Tested
def get_train_val_split(data_idx, data_classes, val_split=0.2):
    """ returns indexes of a split of the dataset in a stratified manner """

    train_idx, valid_idx = train_test_split(data_idx, test_size=val_split, stratify=data_classes)

    return train_idx, valid_idx