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
    RANDOM = "RANDOM"
    BLACKOUT = "BLACKOUT"
    MAINTENANCE = "MAINTENANCE"

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


def mask_data(data: pd.DataFrame, base_mask: np.ndarray, experiment: Experiment, size: float, turbine_count: int | None = None, seed: int = 42):
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
            # mask intervals for each turbine
            for _, turbine_df in data.groupby("TurbID"):
                n_measurements = len(turbine_df)
                # choose random start point
                start = np.random.choice(n_measurements - size)
                # mask `size` consecutive timesteps
                mask[turbine_df.index[start:start+size]] = 0

        case Experiment.MAINTENANCE:
            # size -> consecutive missing intervals => [1, 2, 7, 14] days
            # multiply by 6 * 24 to get number of consecutive entries
            size = int(size * 6 * 24)
            # mask intervals for each turbine
            for _, turbine_df in data.groupby("TurbID"):
                n_measurements = len(turbine_df)
                # choose random start point
                start = np.random.choice(n_measurements - size)
                # mask `size` consecutive timesteps
                mask[turbine_df.index[start:start+size]] = 0

    return mask.astype(bool)


if __name__ == "__main__":
    # for debug
    data = load_data(columns=["TurbID", "P_norm", "datetime"])
    turbines = np.random.choice(data["TurbID"].unique(), 15)
    print(turbines)
    data = data[data["TurbID"].isin(turbines)]
    print(data.index)
