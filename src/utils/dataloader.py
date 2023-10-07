from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
sys.path.append(os.getcwd())

class MusicDataset(Dataset):
    def __init__(self, fpath):
        self.fpath = fpath
        if os.getcwd() not in fpath:
            self.data = torch.load(os.path.join(os.getcwd(), fpath))
        else:
            self.data = torch.load(fpath)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]
    
class CollateFn:
    def __call__(self, batch):
        batch = torch.stack(batch)
        return batch[:, :-1], batch[:, 1:]
    
    
