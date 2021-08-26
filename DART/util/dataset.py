import os
import torch
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

CASIA_REDUCED = "../dataset/casia/casia_clean_align_reduced"
CASIA = "../dataset/casia/casia_webface_clean_align"
LFW = "../dataset/lfw/lfw_align"
LFW_PAIRS = "../dataset/lfw/pairs.txt"

class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

# TODO add other datasets
# Tested
def get_train_dataset(name, reduced=False):
    """ returns only the train dataset """
    name = name.lower()

    train_trans = transforms.Compose ([
        transforms.Resize(128), #  128x128
        transforms.RandomCrop(112), # 
        transforms.RandomHorizontalFlip(), # randomly flipping
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    root = CASIA_REDUCED if reduced else CASIA

    trn_data = ImageFolder(root=root, transform=train_trans)
    input_channels = 3
    input_size = 112

    if reduced:
        n_classes = 2000
    else:
        n_classes = 10575

    # assert statements for classes, input_size
    assert len(trn_data.classes) == n_classes
    assert trn_data[0][0].shape[1] == 112

    return input_size, input_channels, n_classes, trn_data

def get_casia_without_crop(reduced=False):
    trans = transforms.Compose ([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    root = CASIA_REDUCED if reduced else CASIA

    data = ImageFolder(root=root, transform=trans)

    return data

# Tested
def get_train_val_split(data, val_split=0.2):
    """ returns indexes of a split of the dataset in a stratified manner """
    targets = data.targets

    train_idx, valid_idx = train_test_split(np.arange(len(targets)), test_size=val_split, stratify=targets)

    return train_idx, valid_idx

# Tested
def get_lfw_dataset():
    """returns the lfw dataset with path"""

    val_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    lfw_data = ImageFolderWithPaths(LFW, transform=val_trans)

    return lfw_data
