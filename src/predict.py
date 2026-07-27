"""
Inference helpers: batch RUL prediction from saved models, and a
real-time streaming prediction simulation.
"""

import time

import joblib
import numpy as np

from .config import DATASET_NAMES, MODELS_PATH


def load_model_and_scaler(dataset, models_path=MODELS_PATH):
    """Load the saved best model + scaler pair for one dataset."""
    model = joblib.load(f"{models_path}/{dataset}_best_model.pkl")
    scaler = joblib.load(f"{models_path}/{dataset}_scaler.pkl")
    return model, scaler


def load_all_models(dataset_names=DATASET_NAMES, models_path=MODELS_PATH):
    """Load every dataset's saved model + scaler into dicts keyed by dataset."""
    models, scalers = {}, {}
    for dataset in dataset_names:
        models[dataset], scalers[dataset] = load_model_and_scaler(dataset, models_path)
        print(f"{dataset}: model and scaler loaded")
    return models, scalers


def predict_rul(model, scaler, X):
    """
    Scale X with the fitted scaler, predict RUL, and clip negative
    predictions to 0 (RUL cannot be negative in practice).
    """
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    return np.clip(predictions, 0, None)


def predict_all_datasets(split_data, models, scalers, dataset_names=DATASET_NAMES):
    """
    Run predict_rul() for every dataset's held-out X_test and return a
    dict of {dataset: predictions_array}.
    """
    final_predictions = {}

    for dataset in dataset_names:
        X_test = split_data[dataset]["X_test"]
        final_predictions[dataset] = predict_rul(
            models[dataset], scalers[dataset], X_test
        )
        preds = final_predictions[dataset]
        print(f"{dataset}: {len(preds)} predictions, "
              f"min={preds.min():.1f} max={preds.max():.1f}")

    return final_predictions


def simulate_streaming_prediction(model, scaler, X_stream, n_readings=5, delay_seconds=1):
    """
    Simulate sensor readings arriving one cycle at a time and predict
    RUL for each as it comes in. Mirrors what a real-time monitoring
    service would do with a live sensor feed.
    """
    for i in range(min(n_readings, len(X_stream))):
        row = X_stream.iloc[[i]]
        predicted_rul = predict_rul(model, scaler, row)[0]

        print(f"Incoming reading #{i + 1} -> Predicted RUL: {predicted_rul:.1f} cycles")
        time.sleep(delay_seconds)
