"""
preprocessing.py — Final preprocessing before model training.

Handles remaining NaN imputation, label encoding of categorical columns,
and ensures consistent column ordering between train and test.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import TARGET_COL


def preprocess(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
) -> tuple:
    """Preprocess train and test DataFrames for model consumption.

    1. Fill numeric NaNs with column median (fit on train).
    2. Fill categorical NaNs with ``"MISSING"``.
    3. Label-encode remaining string/object columns (fit on combined
       train + test vocabulary to avoid unseen-label errors).
    4. Return aligned numpy arrays.

    Parameters
    ----------
    train_df : pd.DataFrame
        Engineered training data (must contain ``TARGET_COL``).
    test_df : pd.DataFrame
        Engineered test data.
    feature_cols : list[str]
        Ordered list of feature column names.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]
        ``(X_train, y_train, X_test, final_feature_cols)``
    """
    # Ensure we only keep requested features
    missing_train = [c for c in feature_cols if c not in train_df.columns]
    missing_test = [c for c in feature_cols if c not in test_df.columns]
    if missing_train:
        print(f"⚠ Columns missing in train (will be filled with 0): {missing_train}")
        for c in missing_train:
            train_df[c] = 0
    if missing_test:
        print(f"⚠ Columns missing in test (will be filled with 0): {missing_test}")
        for c in missing_test:
            test_df[c] = 0

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].values.copy()

    # ── 1. Numeric NaN → median (fit train, apply test) ──────────────
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    medians = X_train[numeric_cols].median()
    X_train[numeric_cols] = X_train[numeric_cols].fillna(medians)
    X_test[numeric_cols] = X_test[numeric_cols].fillna(medians)

    # ── 2. Categorical NaN → "MISSING" ──────────────────────────────
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        X_train[c] = X_train[c].fillna("MISSING").astype(str)
        X_test[c] = X_test[c].fillna("MISSING").astype(str)

    # ── 3. Label-encode (fit on combined to handle unseen) ───────────
    label_encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[c], X_test[c]], axis=0)
        le.fit(combined)
        X_train[c] = le.transform(X_train[c])
        X_test[c] = le.transform(X_test[c])
        label_encoders[c] = le

    # ── 4. Final safety: fill any remaining NaN with 0 ───────────────
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    final_feature_cols = list(X_train.columns)

    print(f"✓ Preprocessing done — X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"  NaNs in X_train: {X_train.isna().sum().sum()}, X_test: {X_test.isna().sum().sum()}")

    return (
        X_train.values.astype(np.float64),
        y_train.astype(np.float64),
        X_test.values.astype(np.float64),
        final_feature_cols,
    )
