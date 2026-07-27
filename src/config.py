"""
Central configuration for the RUL prediction pipeline.

Keeping paths, column names, and thresholds here means every other
module (and the notebook) can import a single source of truth
instead of redefining constants inline.
"""

import os

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "CMaps")
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results")

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
DATASET_NAMES = ["FD001", "FD002", "FD003", "FD004"]

# NASA C-MAPSS raw column layout: engine id, cycle, 3 operational
# settings, and 21 sensor readings.
BASE_COLUMNS = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
COLUMN_NAMES = BASE_COLUMNS + SENSOR_COLUMNS

# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------
ROLLING_WINDOW_SIZE = 5

# ---------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------
RANDOM_STATE = 42
TRAIN_TEST_SPLIT_RATIO = 0.8
CV_SPLITS = 5

XGB_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 6],
    "learning_rate": [0.01, 0.1],
}

# ---------------------------------------------------------------------
# Maintenance decision thresholds (in predicted RUL cycles)
# ---------------------------------------------------------------------
MAINTENANCE_THRESHOLDS = {
    "Immediate Maintenance Required": 10,
    "Plan Maintenance": 50,
    "Inspection Recommended": 100,
}
