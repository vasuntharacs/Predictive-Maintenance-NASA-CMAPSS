"""
Loading raw NASA C-MAPSS train / test / RUL text files into DataFrames.
"""

import pandas as pd

from .config import COLUMN_NAMES, DATASET_NAMES, DATA_PATH


def load_cmapss_data(dataset_name: str, data_path: str = DATA_PATH):
    """
    Load the train, test, and RUL files for a single C-MAPSS
    sub-dataset (e.g. "FD001").

    Returns
    -------
    train_data, test_data, rul_data : pandas.DataFrame
    """
    train_file = f"{data_path}/train_{dataset_name}.txt"
    test_file = f"{data_path}/test_{dataset_name}.txt"
    rul_file = f"{data_path}/RUL_{dataset_name}.txt"

    train_data = pd.read_csv(train_file, sep=r"\s+", header=None)
    test_data = pd.read_csv(test_file, sep=r"\s+", header=None)
    rul_data = pd.read_csv(rul_file, sep=r"\s+", header=None)

    train_data.columns = COLUMN_NAMES
    test_data.columns = COLUMN_NAMES

    return train_data, test_data, rul_data


def load_all_datasets(dataset_names=DATASET_NAMES, data_path: str = DATA_PATH):
    """
    Load every sub-dataset into a single nested dict:
    {"FD001": {"train": df, "test": df, "rul": df}, ...}
    """
    data = {}

    for dataset in dataset_names:
        train, test, rul = load_cmapss_data(dataset, data_path)
        data[dataset] = {"train": train, "test": test, "rul": rul}
        print(f"{dataset} | train: {train.shape}  test: {test.shape}  rul: {rul.shape}")

    return data
