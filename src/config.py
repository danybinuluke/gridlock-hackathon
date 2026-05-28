"""
config.py — Central configuration for the traffic demand prediction pipeline.

All tunable constants, file paths, hyperparameters, and feature definitions
are stored here so every module draws from a single source of truth.
"""

import os

# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "submissions")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
EDA_DIR = os.path.join(OUTPUT_DIR, "eda")

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

# Ensure output directories exist
for _d in [OUTPUT_DIR, MODEL_DIR, EDA_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# General
# ─────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_FOLDS = 5
TARGET_COL = "demand"
INDEX_COL = "Index"

# ─────────────────────────────────────────────────────────────────────
# Peak / Night hour ranges
# ─────────────────────────────────────────────────────────────────────
PEAK_HOURS = [7, 8, 9, 17, 18, 19, 20]       # morning + evening rush
NIGHT_HOURS = [22, 23, 0, 1, 2, 3, 4]

# ─────────────────────────────────────────────────────────────────────
# Target-encoding columns
# ─────────────────────────────────────────────────────────────────────
TARGET_ENCODE_COLS = [
    "geohash", "geohash_time_slot", "geohash_hour",
    "geohash_prefix_4", "geohash_prefix_5",
    "Weather", "RoadType", "weekday_hour",
    "roadtype_hour", "roadtype_lanes", "temp_hour"
]

# ─────────────────────────────────────────────────────────────────────
# Model hyperparameters
# ─────────────────────────────────────────────────────────────────────
CATBOOST_PARAMS = {
    "iterations": 6000,
    "learning_rate": 0.02,
    "depth": 10,
    "l2_leaf_reg": 2,
    "random_seed": RANDOM_SEED,
    "verbose": 200,
    "loss_function": "RMSE",
    "eval_metric": "R2",
    "task_type": "CPU",
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.3,
    "min_data_in_leaf": 5,
}

LGBM_PARAMS = {
    "n_estimators": 6000,
    "learning_rate": 0.02,
    "num_leaves": 511,
    "max_depth": -1,
    "min_child_samples": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.01,
    "reg_lambda": 0.1,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

XGB_PARAMS = {
    "n_estimators": 6000,
    "learning_rate": 0.02,
    "max_depth": 10,
    "min_child_weight": 2,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.01,
    "reg_lambda": 0.1,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": 0,
    "tree_method": "hist",
}
