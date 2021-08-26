from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.sampler import RandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
import torchvision.transforms as trans

class Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset

    """
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        im, labels = self.dataset[self.indices[idx]]
        return self.transform(im), labels, self.indices[idx]

    def __len__(self):
        return len(self.indices)

train_trans = trans.Compose ([
        trans.ToPILImage(),
        trans.RandomResizedCrop(112),
        trans.RandomVerticalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_trans = trans.Compose([
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

data = ImageFolder(root="../../dataset/casia/casia_webface_clean_align", transform=trans.ToTensor())

targets = data.targets

train_idx, valid_idx = train_test_split(np.arange(len(targets)), test_size=0.1, shuffle=True, stratify=targets)

print(data[0][0].shape)

# --------------------------------------- Subsets with different transformations but not sure whether it is correct
train_data = Subset(data, train_idx, trans.Compose([]))

val_data = Subset(data, valid_idx, val_trans)

train_loader = DataLoader(
    train_data, 
    batch_size= 64, 
    sampler=RandomSampler(train_data),
    pin_memory=True
)

val_loader = DataLoader(
    val_data,
    batch_size=16,
    sampler=RandomSampler(val_data),
    pin_memory=True
)

#---------------------------------------------- Just use random subset sampler
train_loader = DataLoader(
    data,
    batch_size=64,
    sampler=SubsetRandomSampler(train_idx),
    pin_memory=True
)
