from torch.utils.data import Dataset

class WindFarmDataset(Dataset):
    def __init__(self, X, y):
        if len(X) != len(y):
            raise AssertionError(f"data and target length mismatch: {len(X)} != {len(y)}")
        self.data = [{"X": X_one, "y": y_one} for X_one, y_one in zip(X, y)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample