import os
import random
import torch
import numpy as np
import pandas as pd
from enum import Enum

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class Experiment(Enum):
    RANDOM = "random"
    BLACKOUT = "blackout"
    MAINTENANCE = "maintenance"

def load_data(columns: list | None = None):
    processed_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/processed_data.csv"))

    int_columns = processed_df.select_dtypes(include=['int64']).columns.tolist()
    processed_df[int_columns] = processed_df[int_columns].apply(lambda arg: pd.to_numeric(arg, downcast='integer'))

    float_columns = processed_df.select_dtypes(include=['float64']).columns.tolist()
    processed_df[float_columns] = processed_df[float_columns].apply(lambda arg: pd.to_numeric(arg, downcast='float'))

    processed_df['datetime'] = pd.to_datetime(processed_df['datetime']).astype('datetime64[s]')

    if columns is not None:
        processed_df = processed_df[columns]

    return processed_df


def split_data(df: pd.DataFrame, splits: list | None = None):
    if splits is None:
        return df
    if not isinstance(splits, list):
        raise ValueError("Splits are not a list")
    if len(splits) != 3 and sum(splits) > 1:
        raise ValueError("Invalid splits")
    timestamps = df["datetime"].unique()
    timestamps = timestamps[timestamps.argsort()]

    split_sizes = [int(split * len(timestamps)) for split in splits]
    train_times, val_times, test_times = np.split(timestamps, [split_sizes[0], split_sizes[0] + split_sizes[1]])
    
    train = df[df["datetime"].isin(train_times)]
    val = df[df["datetime"].isin(val_times)]
    test = df[df["datetime"].isin(test_times)]
    return train, val, test


def mask_data(data: pd.DataFrame, base_mask: np.ndarray, experiment: Experiment, size: float, fraction: float | None = None, turbine_count: int | None = None, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    """
        Note: at this point it's assumed that data is sorted by timestep (done in `load_data()`)
    """

    mask = base_mask
    if mask is None:
        mask = np.ones(len(data), dtype=bool)
        data = data.reset_index()

    if turbine_count is not None:
        # If turbine count provided, sample turbine_count random turbines for masking
        turbines = np.random.choice(data["TurbID"].unique(), turbine_count)
        data = data[data["TurbID"].isin(turbines)]

    match experiment:
        case Experiment.RANDOM:
            # size -> absolute fraction of missing values => [1, 2, 5, 10] %
            mask_size = int(len(data) * size)
            mask[np.random.choice(data.index, mask_size)] = 0
        case Experiment.BLACKOUT:
            # size -> consecutive missing intervals => [30, 60, 150, 300] minutes
            # divide by 10 to get number of consecutive entries
            size = int(size / 10)
            # ensure minimal spacing between intervals
            min_spacing = int(np.ceil(size * 0.5))

            # fraction should be provided
            if fraction is None:
                raise EOFError("fraction cannot be None for blackout experiment")
            
            turb_groups = list(data.groupby("TurbID"))

            # n - number of masks needed. Sample until exceeded
            n = int(len(data) * fraction / size)
            masked_count = 0
            while masked_count < n:
                turb_id, group = turb_groups[np.random.randint(len(turb_groups))]

                idxs = group.index.to_numpy()
                unmasked = np.where(mask[idxs] == 1)[0]
                
                if len(unmasked) < size:
                    continue
                
                candidate_starts = [
                    i for i in unmasked
                    if i + size <= len(group)
                    and np.all( mask[idxs[ max(0, i - min_spacing) : min(i + size + min_spacing, len(group)) ]] )
                ]

                if not candidate_starts:
                    print(candidate_starts)
                    continue

                start = np.random.choice(candidate_starts)
                blackout_idxs = idxs[start : start + size]
                mask[blackout_idxs] = 0
                masked_count += 1

        case Experiment.MAINTENANCE:
            # size -> consecutive missing intervals => [1, 2, 7, 14] days
            # multiply by 6 * 24 to get number of consecutive entries
            size = int(size * 6 * 24)
            # ensure minimal spacing between intervals
            min_spacing = int(np.ceil(size * 0.5))

            if fraction is None:
                raise EOFError("fraction cannot be None for maintenance experiment")
            
            turb_groups = list(data.groupby("TurbID"))

            # n - number of masks needed. Sample until exceeded
            n = int(len(data) * fraction / size)
            masked_count = 0
            while masked_count < n:
                turb_id, group = turb_groups[np.random.randint(len(turb_groups))]

                idxs = group.index.to_numpy()
                unmasked = np.where(mask[idxs] == 1)[0]
                
                if len(unmasked) < size:
                    continue
                
                candidate_starts = [
                    i for i in unmasked
                    if i + size <= len(group)
                    and np.all( mask[idxs[ max(0, i - min_spacing) : min(i + size + min_spacing, len(group)) ]] )
                ]

                if not candidate_starts:
                    continue

                start = np.random.choice(candidate_starts)
                blackout_idxs = idxs[start : start + size]
                mask[blackout_idxs] = 0
                masked_count += 1

    return mask.astype(bool)


if __name__ == "__main__":
    # for debug
    data = load_data(columns=["TurbID", "Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv", "Patv", "datetime"])

    turbines_idx = [9, 10, 11, 12, 31, 32, 33, 34, 35, 52, 53, 54, 55, 56, 57]
    data = data[data["TurbID"].isin(turbines_idx)]
    train_data, val_data, test_data = split_data(data, splits=[0.7, 0.2, 0.1])

    train_data.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    print(mask_data(train_data, base_mask=None, experiment=Experiment.MAINTENANCE, size = 14, fraction=0.02))
