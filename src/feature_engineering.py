"""
Degradation-trend feature engineering and automated feature selection.
"""

from sklearn.feature_selection import SelectKBest, f_regression

from .config import DATASET_NAMES, ROLLING_WINDOW_SIZE


def add_rolling_features(df, sensor_cols, window_size=ROLLING_WINDOW_SIZE):
    """
    Add, per sensor, a rolling mean, rolling std, and cycle-to-cycle
    diff computed within each engine. These engineered features
    capture the degradation trend rather than a single noisy reading.
    """
    df = df.sort_values(["engine_id", "cycle"]).copy()

    for sensor in sensor_cols:
        grouped = df.groupby("engine_id")[sensor]

        df[f"{sensor}_roll_mean"] = grouped.transform(
            lambda x: x.rolling(window_size, min_periods=1).mean()
        )
        df[f"{sensor}_roll_std"] = grouped.transform(
            lambda x: x.rolling(window_size, min_periods=1).std().fillna(0)
        )
        df[f"{sensor}_diff"] = grouped.diff().fillna(0)

    return df


def engineer_features_for_all(clean_data, dataset_names=DATASET_NAMES,
                               window_size=ROLLING_WINDOW_SIZE):
    """Apply add_rolling_features() to train/test of every dataset."""
    for dataset in dataset_names:
        train_data = clean_data[dataset]["train"]
        test_data = clean_data[dataset]["test"]

        sensor_cols = [col for col in train_data.columns if "sensor" in col]

        clean_data[dataset]["train"] = add_rolling_features(
            train_data, sensor_cols, window_size
        )
        clean_data[dataset]["test"] = add_rolling_features(
            test_data, sensor_cols, window_size
        )

        print(f"{dataset}: shape after feature engineering "
              f"{clean_data[dataset]['train'].shape}")

    return clean_data


def prepare_features_and_target(clean_data, dataset_names=DATASET_NAMES):
    """
    Split each dataset's cleaned training frame into X (features) and
    y (RUL target).
    """
    prepared_data = {}

    for dataset in dataset_names:
        train_data = clean_data[dataset]["train"]

        X = train_data.drop(columns=["RUL"])
        y = train_data["RUL"]

        prepared_data[dataset] = {"X": X, "y": y}
        print(f"{dataset}: X {X.shape}  y {y.shape}")

    return prepared_data


def select_top_features(X_train, y_train, k=10):
    """
    Run SelectKBest (f_regression) to identify the k most predictive
    columns. Returns the list of selected feature names.
    """
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X_train, y_train)
    return list(X_train.columns[selector.get_support()])
