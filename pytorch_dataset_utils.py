import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyDataset(Dataset):
    """create [MyDataset] dataset"""
    def __init__(self):
        super().__init__()
        
        self.data = []
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # self.data[idx]
        
        return {
            # "key" : torch.tensor(..., dtype=...),
            # "key" : torch.tensor(..., dtype=...)
        }
        
"""
Example of creating train and ptest datasets.

train_dataset = MyDataset(
    # params1=...,
    # params2=...,
)
train_dataset = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=False,  # should be carefully set
    num_workers=5
)

ptest_dataset = MyDataset(
    # params1=...,
    # params2=...,
)
ptest_dataset = DataLoader(
    ptest_dataset,
    batch_size=32,
    shuffle=False,  # should be carefully set
    num_workers=5
)
"""

"""
Example of create train, valid, ptest datasets.

train_dataset = MyDataset(
    # params1=...,
    # params2=...,
)
train_dataset = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=False,  # should be carefully set
    num_workers=5
)

valid_dataset = MyDataset(
    # params1=...,
    # params2=...,
)
valid_dataset = DataLoader(
    valid_dataset,
    batch_size=32,
    shuffle=False,  # should be carefully set
    num_workers=5
)

ptest_dataset = MyDataset(
    # params1=...,
    # params2=...,
)
ptest_dataset = DataLoader(
    ptest_dataset,
    batch_size=32,
    shuffle=False,  # should be carefully set
    num_workers=5
)
"""

"""
Example of create train, valid, ptest datasets in the second way.

train_dataset = MyDataset(
    # params1=...,
    # params2=...,
)

indices = [i for i in range(len(train_dataset))]
# [need]: import random
random.shuffle(indices)

validset_size = int(0.1 * len(indices))
validset_indices = indices[:validset_size]
trainset_indices = indices[validset_size:]

# [need]: from torch.utils.data import Subset
valid_dataset = DataLoader(
    # should be created first because of train dataset
    Subset(train_dataset, validset_indices),
    batch_size=32,
    shuffle=False,  # should be carefully set
    num_workers=5
)

train_dataset = DataLoader(
    Subset(train_dataset, trainset_indices),
    batch_size=32,
    shuffle=False,  # should be carefully set
    num_workers=5
)

ptest_dataset = MyDataset(
    # params1=...,
    # params2=...,
)
ptest_dataset = DataLoader(
    ptest_dataset,
    batch_size=32,
    shuffle=False,  # should be carefully set
    num_workers=5
)
"""
