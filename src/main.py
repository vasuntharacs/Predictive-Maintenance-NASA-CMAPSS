"""
End-to-end pipeline runner.

Usage:
    python -m src.main

Runs the full flow the notebook walks through interactively: load data,
calculate RUL, clean sensors, engineer features, split/scale, train and
tune models, save them, predict, and build maintenance reports.
"""

import warnings

from . import config
from .data_loader import load_all_datasets
from .evaluate import optimized_results_to_dataframe, results_to_dataframe
from .feature_engineering import engineer_features_for_all, prepare_features_and_target
from .maintenance import build_all_reports, summarize_reports
from .predict import predict_all_datasets
from .preprocessing import add_rul_to_all, clean_missing_and_outliers, remove_constant_sensors
from .train import (
    save_models,
    scale_all_datasets,
    split_all_datasets,
    time_series_cross_validate,
    train_and_compare_models,
    tune_hyperparameters,
)

warnings.filterwarnings("ignore")


def run_pipeline():
    print("Step 1/8: Loading datasets")
    data = load_all_datasets()

    print("\nStep 2/8: Calculating RUL")
    data = add_rul_to_all(data)

    print("\nStep 3/8: Removing constant sensors")
    clean_data = remove_constant_sensors(data)

    print("\nStep 4/8: Handling missing values and outliers")
    clean_data = clean_missing_and_outliers(clean_data)

    print("\nStep 5/8: Engineering rolling / diff features")
    clean_data = engineer_features_for_all(clean_data)
    prepared_data = prepare_features_and_target(clean_data)

    print("\nStep 6/8: Splitting and scaling")
    split_data = split_all_datasets(prepared_data)
    scaled_data = scale_all_datasets(split_data)

    print("\nStep 7/8: Training, cross-validating, and tuning models")
    model_results, _ = train_and_compare_models(scaled_data)
    print(results_to_dataframe(model_results))

    time_series_cross_validate(scaled_data)

    optimized_results, best_models = tune_hyperparameters(scaled_data)
    print(optimized_results_to_dataframe(optimized_results))

    print("\nStep 8/8: Saving models, predicting, and building maintenance reports")
    scalers = {ds: scaled_data[ds]["scaler"] for ds in config.DATASET_NAMES}
    save_models(best_models, scalers)

    final_predictions = predict_all_datasets(split_data, best_models, scalers)
    maintenance_reports = build_all_reports(final_predictions)
    summary_df = summarize_reports(maintenance_reports)
    print(summary_df)

    return {
        "model_results": model_results,
        "optimized_results": optimized_results,
        "best_models": best_models,
        "maintenance_reports": maintenance_reports,
        "summary": summary_df,
    }


if __name__ == "__main__":
    run_pipeline()
