"""
Shared regression evaluation metrics and result-table helpers.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    """Return MAE, RMSE, and R2 for a set of predictions as a dict."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "R2 Score": r2}


def results_to_dataframe(model_results):
    """
    Flatten the nested {dataset: {model_name: metrics}} dict produced
    by train_and_compare_models() into a tidy comparison DataFrame.
    """
    rows = []
    for dataset, models in model_results.items():
        for model_name, metrics in models.items():
            rows.append({"Dataset": dataset, "Model": model_name, **metrics})

    return pd.DataFrame(rows)


def optimized_results_to_dataframe(optimized_results):
    """
    Flatten the {dataset: {..., "MAE":..., "RMSE":..., "R2 Score":...}}
    dict produced by tune_hyperparameters() into a DataFrame (drops
    the best-params dict, which doesn't belong in a metrics table).
    """
    rows = [
        {
            "Dataset": dataset,
            "MAE": values["MAE"],
            "RMSE": values["RMSE"],
            "R2 Score": values["R2 Score"],
        }
        for dataset, values in optimized_results.items()
    ]
    return pd.DataFrame(rows)
