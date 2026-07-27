"""
Cleaning steps applied before feature engineering:
- RUL target calculation
- constant-sensor removal
- missing value / outlier handling
"""

from .config import DATASET_NAMES


def calculate_rul(train_data):
    """
    Add a RUL column to a single training DataFrame: for every row,
    RUL = (that engine's max observed cycle) - (current cycle).
    """
    train_data = train_data.copy()
    max_cycle = train_data.groupby("engine_id")["cycle"].max()

    train_data["RUL"] = train_data.apply(
        lambda row: max_cycle[row["engine_id"]] - row["cycle"], axis=1
    )
    return train_data


def add_rul_to_all(data, dataset_names=DATASET_NAMES):
    """Apply calculate_rul() to every dataset's training split in place."""
    for dataset in dataset_names:
        data[dataset]["train"] = calculate_rul(data[dataset]["train"])
        print(f"{dataset}: RUL calculation completed")
    return data


def find_constant_sensors(train_data):
    """Return sensor columns that never change value (carry no signal)."""
    sensor_columns = [col for col in train_data.columns if "sensor" in col]
    return [s for s in sensor_columns if train_data[s].nunique() == 1]


def remove_constant_sensors(data, dataset_names=DATASET_NAMES):
    """
    Drop constant sensors from both train and test splits of every
    dataset. Returns a new dict keyed the same way as `data`.
    """
    clean_data = {}

    for dataset in dataset_names:
        train_data = data[dataset]["train"].copy()
        test_data = data[dataset]["test"].copy()

        constant_sensors = find_constant_sensors(train_data)

        train_data = train_data.drop(columns=constant_sensors)
        test_data = test_data.drop(columns=constant_sensors)

        clean_data[dataset] = {"train": train_data, "test": test_data}
        print(f"{dataset}: removed {constant_sensors} -> new shape {train_data.shape}")

    return clean_data


def handle_missing_and_outliers(df, sensor_cols):
    """
    Forward/backward fill missing sensor readings within each engine,
    then cap outliers per sensor using a wide (3x) IQR fence.

    The C-MAPSS data itself is clean; this exists so the same pipeline
    can be pointed at real, noisier plant sensor feeds.
    """
    df = df.copy()

    df[sensor_cols] = df.groupby("engine_id")[sensor_cols].transform(
        lambda x: x.ffill().bfill()
    )

    for col in sensor_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        df[col] = df[col].clip(lower, upper)

    return df


def clean_missing_and_outliers(clean_data, dataset_names=DATASET_NAMES):
    """Apply handle_missing_and_outliers() to train/test of every dataset."""
    for dataset in dataset_names:
        train_data = clean_data[dataset]["train"]
        test_data = clean_data[dataset]["test"]

        sensor_cols = [col for col in train_data.columns if "sensor" in col]

        clean_data[dataset]["train"] = handle_missing_and_outliers(train_data, sensor_cols)
        clean_data[dataset]["test"] = handle_missing_and_outliers(test_data, sensor_cols)

        print(f"{dataset}: missing/outlier handling applied")

    return clean_data
