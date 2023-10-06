from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
sys.path.append(os.getcwd())

class MusicDataset(Dataset):
    def __init__(self, fpath):
        self.fpath = fpath
        self.listdir = os.listdir(self.fpath)

    def __getitem__(self, index):
        self.data = torch.load(os.path.join(self.fpath, self.listdir[index]))
        return self.data[index]

    def __len__(self):
        return len(self.listdir)
    
class CollateFn:
    def __call__(self, batch):
        return batch[:, :-1], batch[:, 1:]
    
    
