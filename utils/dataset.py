from typing import NamedTuple
import torch
from torch.utils.data import Dataset

class Item(NamedTuple):
    X: torch.Tensor
    y: torch.Tensor

class WindFarmDataset(Dataset):
    def __init__(self, X, y):
        if len(X) != len(y):
            raise AssertionError(f"data and target length mismatch: {len(X)} != {len(y)}")
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return Item(X=self.X[idx], y=self.y[idx])