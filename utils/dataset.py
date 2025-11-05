from typing import NamedTuple
import torch
from torch.utils.data import Dataset

class Item(NamedTuple):
    X: torch.Tensor
    y: torch.Tensor
    nan_mask: torch.Tensor
    data_mask: torch.Tensor

class WindFarmDataset(Dataset):
    def __init__(
            self,
            X: torch.Tensor,
            y: torch.Tensor,
            nan_mask: torch.Tensor,
            data_mask: torch.Tensor,
        ):
        if len(X) != len(y):
            raise AssertionError(f"data and target length mismatch: {len(X)} != {len(y)}")
        self.X = X
        self.y = y
        self.nan_mask = nan_mask
        self.data_mask = data_mask

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return Item(
            X=self.X[idx],
            y=self.y[idx],
            nan_mask=self.nan_mask[idx],
            data_mask=self.data_mask[idx],
        )