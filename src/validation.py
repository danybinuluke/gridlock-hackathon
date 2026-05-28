"""
validation.py — Cross-validation strategy for demand prediction.

Uses KFold for competition scoring.  The train/test split is already
temporal (train=day48+early49, test=rest of day49), so within-train
cross-validation uses KFold to maximise each fold's training size.
"""

import numpy as np
from sklearn.model_selection import KFold

from config import N_FOLDS, RANDOM_SEED


def get_cv_splits(X, y, timestamps=None, n_folds: int = N_FOLDS) -> list:
    """Generate cross-validation splits.

    The competition already enforces a temporal train/test boundary
    (train = day 48 + early day 49; test = rest of day 49).  Within
    the training set we use KFold to give each fold maximum data.

    Random splits are generally invalid for time-series because demand
    has temporal autocorrelation.  However, since the competition
    train/test boundary is already temporal and we have only 2 days
    of data, using TimeSeriesSplit *within* training cripples fold 0
    (it would train on only ~12K rows).  KFold gives each fold ~62K
    training rows, producing far stronger models.

    Parameters
    ----------
    X : array-like
        Feature matrix (used only for its length).
    y : array-like
        Target vector.
    timestamps : array-like, optional
        Not used in KFold mode but kept for API compatibility.
    n_folds : int
        Number of folds (default from config).

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of ``(train_indices, val_indices)`` tuples.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    splits = []
    for train_idx, val_idx in kf.split(X):
        splits.append((train_idx, val_idx))

    print(f"[OK] Created {n_folds} KFold CV splits")
    for i, (tr, va) in enumerate(splits):
        print(f"  Fold {i}: train={len(tr):,}, val={len(va):,}")

    return splits
