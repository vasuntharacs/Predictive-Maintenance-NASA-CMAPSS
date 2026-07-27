"""
Time-ordered train/test splitting, scaling, model comparison,
time-series cross-validation, and hyperparameter tuning.
"""

import os

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .config import (
    CV_SPLITS,
    DATASET_NAMES,
    MODELS_PATH,
    RANDOM_STATE,
    TRAIN_TEST_SPLIT_RATIO,
    XGB_PARAM_GRID,
)
from .evaluate import regression_metrics


def time_ordered_split(X, y, split_ratio=TRAIN_TEST_SPLIT_RATIO):
    """
    Sort by engine_id/cycle and take the first `split_ratio` fraction
    as train, the rest as test. This avoids shuffling time-series data
    across the split, which would leak future cycles into training.
    """
    sorted_index = X.sort_values(["engine_id", "cycle"]).index
    X = X.loc[sorted_index]
    y = y.loc[sorted_index]

    split_point = int(len(X) * split_ratio)

    return (
        X.iloc[:split_point], X.iloc[split_point:],
        y.iloc[:split_point], y.iloc[split_point:],
    )


def split_all_datasets(prepared_data, dataset_names=DATASET_NAMES,
                        split_ratio=TRAIN_TEST_SPLIT_RATIO):
    """Apply time_ordered_split() across every dataset."""
    split_data = {}

    for dataset in dataset_names:
        X = prepared_data[dataset]["X"]
        y = prepared_data[dataset]["y"]

        X_train, X_test, y_train, y_test = time_ordered_split(X, y, split_ratio)

        split_data[dataset] = {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
        }
        print(f"{dataset}: X_train {X_train.shape}  X_test {X_test.shape}")

    return split_data


def scale_all_datasets(split_data, dataset_names=DATASET_NAMES):
    """
    Fit a StandardScaler on each dataset's training split only, then
    transform both train and test. Returns scaled arrays plus the
    fitted scaler (needed later for inference).
    """
    scaled_data = {}

    for dataset in dataset_names:
        X_train = split_data[dataset]["X_train"]
        X_test = split_data[dataset]["X_test"]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        scaled_data[dataset] = {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": split_data[dataset]["y_train"],
            "y_test": split_data[dataset]["y_test"],
            "scaler": scaler,
        }
        print(f"{dataset}: scaled X_train {X_train_scaled.shape}")

    return scaled_data


def get_candidate_models(random_state=RANDOM_STATE):
    """The three model families compared in the notebook."""
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
        "XGBoost": XGBRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state
        ),
    }


def train_and_compare_models(scaled_data, dataset_names=DATASET_NAMES):
    """
    Train each candidate model per dataset and score it on the held-out
    time-ordered test split. Returns (model_results, trained_models).
    """
    model_results = {}
    trained_models = {}

    for dataset in dataset_names:
        X_train = scaled_data[dataset]["X_train"]
        X_test = scaled_data[dataset]["X_test"]
        y_train = scaled_data[dataset]["y_train"]
        y_test = scaled_data[dataset]["y_test"]

        dataset_results = {}
        dataset_models = {}

        for name, model in get_candidate_models().items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            dataset_results[name] = regression_metrics(y_test, y_pred)
            dataset_models[name] = model

        model_results[dataset] = dataset_results
        trained_models[dataset] = dataset_models
        print(f"{dataset}: trained {list(dataset_results.keys())}")

    return model_results, trained_models


def time_series_cross_validate(scaled_data, dataset_names=DATASET_NAMES,
                                n_splits=CV_SPLITS, random_state=RANDOM_STATE):
    """
    5-fold (default) TimeSeriesSplit cross-validation of an XGBoost
    model per dataset, on the (already time-ordered) training split.
    """
    cv_results = {}

    for dataset in dataset_names:
        X = scaled_data[dataset]["X_train"]
        y = scaled_data[dataset]["y_train"]

        model = XGBRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state
        )
        tscv = TimeSeriesSplit(n_splits=n_splits)

        mae_scores, rmse_scores, r2_scores = [], [], []

        for train_idx, val_idx in tscv.split(X):
            model.fit(X[train_idx], y.iloc[train_idx])
            y_pred = model.predict(X[val_idx])

            metrics = regression_metrics(y.iloc[val_idx], y_pred)
            mae_scores.append(metrics["MAE"])
            rmse_scores.append(metrics["RMSE"])
            r2_scores.append(metrics["R2 Score"])

        cv_results[dataset] = {
            "MAE": float(np.mean(mae_scores)),
            "RMSE": float(np.mean(rmse_scores)),
            "R2 Score": float(np.mean(r2_scores)),
        }
        print(f"{dataset}: CV MAE={cv_results[dataset]['MAE']:.4f} "
              f"RMSE={cv_results[dataset]['RMSE']:.4f} "
              f"R2={cv_results[dataset]['R2 Score']:.4f}")

    return cv_results


def tune_hyperparameters(scaled_data, dataset_names=DATASET_NAMES,
                          param_grid=XGB_PARAM_GRID, random_state=RANDOM_STATE):
    """
    GridSearchCV over an XGBoost model per dataset. Returns
    (optimized_results, best_models).
    """
    optimized_results = {}
    best_models = {}

    for dataset in dataset_names:
        X_train = scaled_data[dataset]["X_train"]
        y_train = scaled_data[dataset]["y_train"]
        X_test = scaled_data[dataset]["X_test"]
        y_test = scaled_data[dataset]["y_test"]

        xgb = XGBRegressor(random_state=random_state)
        grid_search = GridSearchCV(
            estimator=xgb, param_grid=param_grid, cv=3,
            scoring="neg_mean_squared_error", n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_models[dataset] = best_model

        y_pred = best_model.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)

        optimized_results[dataset] = {
            "Best Parameters": grid_search.best_params_,
            **metrics,
        }
        print(f"{dataset}: best params {grid_search.best_params_} -> {metrics}")

    return optimized_results, best_models


def save_models(best_models, scalers, models_path=MODELS_PATH,
                 dataset_names=DATASET_NAMES):
    """
    Persist each dataset's best model and fitted scaler to disk with
    joblib, as `<DATASET>_best_model.pkl` / `<DATASET>_scaler.pkl`.
    """
    os.makedirs(models_path, exist_ok=True)

    for dataset in dataset_names:
        model_path = os.path.join(models_path, f"{dataset}_best_model.pkl")
        scaler_path = os.path.join(models_path, f"{dataset}_scaler.pkl")

        joblib.dump(best_models[dataset], model_path)
        joblib.dump(scalers[dataset], scaler_path)

        print(f"{dataset}: saved model -> {model_path}, scaler -> {scaler_path}")
