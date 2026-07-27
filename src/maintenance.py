"""
Translate predicted RUL values into actionable maintenance categories
and per-dataset / fleet-wide reports.
"""

import pandas as pd

from .config import DATASET_NAMES, MAINTENANCE_THRESHOLDS


def maintenance_status(rul):
    """
    Map a single predicted RUL (in cycles) to a maintenance action
    category, using the cutoffs defined in config.MAINTENANCE_THRESHOLDS.
    """
    if rul <= MAINTENANCE_THRESHOLDS["Immediate Maintenance Required"]:
        return "Immediate Maintenance Required"
    elif rul <= MAINTENANCE_THRESHOLDS["Plan Maintenance"]:
        return "Plan Maintenance"
    elif rul <= MAINTENANCE_THRESHOLDS["Inspection Recommended"]:
        return "Inspection Recommended"
    else:
        return "Normal - Continue Monitoring"


def build_maintenance_report(predictions):
    """
    Given an array of predicted RUL values for one dataset (one row
    per engine), return a DataFrame with an Engine_ID, Predicted_RUL,
    and Maintenance_Status column.
    """
    report = pd.DataFrame({
        "Engine_ID": range(1, len(predictions) + 1),
        "Predicted_RUL": predictions,
    })
    report["Maintenance_Status"] = report["Predicted_RUL"].apply(maintenance_status)
    return report


def build_all_reports(final_predictions, dataset_names=DATASET_NAMES):
    """Apply build_maintenance_report() to every dataset's predictions."""
    reports = {}
    for dataset in dataset_names:
        reports[dataset] = build_maintenance_report(final_predictions[dataset])
        print(f"{dataset}: maintenance report built ({len(reports[dataset])} engines)")
    return reports


def summarize_reports(maintenance_reports, dataset_names=DATASET_NAMES):
    """
    Roll up per-dataset maintenance reports into one fleet-wide summary
    of engine counts per maintenance status.
    """
    summary = []

    for dataset in dataset_names:
        status_counts = maintenance_reports[dataset]["Maintenance_Status"].value_counts()

        for status, count in status_counts.items():
            summary.append({
                "Dataset": dataset,
                "Maintenance_Status": status,
                "Number_of_Engines": count,
            })

    return pd.DataFrame(summary)
