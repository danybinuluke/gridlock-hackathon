"""
models.py — OOF training of CatBoost, LightGBM, and XGBoost regressors.

Each model is trained with early stopping across K folds.
Out-of-fold (OOF) predictions are collected for stacking / ensemble
weight optimisation, and per-fold test predictions are averaged.

NO target transform — GBDT handles skewed targets natively,
and power transforms amplify prediction errors on inverse.
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import r2_score

from config import (
    CATBOOST_PARAMS, LGBM_PARAMS, XGB_PARAMS,
    MODEL_DIR, OUTPUT_DIR, RANDOM_SEED,
)

warnings.filterwarnings("ignore")


# =====================================================================
#  Main OOF training
# =====================================================================
def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    cv_splits: list,
    feature_names: list,
) -> dict:
    """Train CatBoost, LightGBM, and XGBoost with OOF strategy.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix (day-49 rows only).
    y_train : np.ndarray
        Training target vector.
    X_test : np.ndarray
        Test feature matrix.
    cv_splits : list[tuple]
        List of ``(train_idx, val_idx)`` from ``get_cv_splits``.
    feature_names : list[str]
        Column names (used for importance plots).

    Returns
    -------
    dict
        ``{model_name: (oof_preds, test_preds_avg, fold_scores)}``
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_folds = len(cv_splits)
    results = {}

    # ── CatBoost ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING — CatBoost")
    print("=" * 60)
    results["catboost"] = _train_catboost(
        X_train, y_train, X_test, cv_splits, feature_names, n_train, n_test, n_folds
    )

    # ── LightGBM ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING — LightGBM")
    print("=" * 60)
    results["lgbm"] = _train_lgbm(
        X_train, y_train, X_test, cv_splits, feature_names, n_train, n_test, n_folds
    )

    # ── XGBoost ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING — XGBoost")
    print("=" * 60)
    results["xgb"] = _train_xgb(
        X_train, y_train, X_test, cv_splits, feature_names, n_train, n_test, n_folds
    )

    return results


# =====================================================================
#  CatBoost
# =====================================================================
def _train_catboost(X_train, y_train, X_test, cv_splits, feature_names,
                    n_train, n_test, n_folds):
    """Train CatBoost with OOF and return (oof, test_avg, scores)."""
    from catboost import CatBoostRegressor, Pool

    oof = np.zeros(n_train)
    test_preds = np.zeros(n_test)
    scores = []
    importances = np.zeros(len(feature_names))

    for fold, (tr_idx, va_idx) in enumerate(cv_splits):
        print(f"\n  Fold {fold} ...")
        train_pool = Pool(X_train[tr_idx], y_train[tr_idx], feature_names=feature_names)
        val_pool = Pool(X_train[va_idx], y_train[va_idx], feature_names=feature_names)

        params = CATBOOST_PARAMS.copy()
        model = CatBoostRegressor(**params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=50,
            verbose=200,
        )

        oof[va_idx] = model.predict(X_train[va_idx])
        test_preds += model.predict(X_test) / n_folds
        fold_r2 = r2_score(y_train[va_idx], oof[va_idx])
        scores.append(fold_r2)
        importances += model.get_feature_importance() / n_folds
        print(f"  Fold {fold} R2: {fold_r2:.6f}")

        joblib.dump(model, os.path.join(MODEL_DIR, f"catboost_fold{fold}.pkl"))

    overall = r2_score(y_train, oof)
    print(f"\n  CatBoost OOF R2: {overall:.6f}  (folds: {np.mean(scores):.6f} +/- {np.std(scores):.6f})")

    _save_importance(feature_names, importances, "catboost")
    return oof, test_preds, scores


# =====================================================================
#  LightGBM
# =====================================================================
def _train_lgbm(X_train, y_train, X_test, cv_splits, feature_names,
                n_train, n_test, n_folds, seed=None):
    """Train LightGBM with OOF and return (oof, test_avg, scores)."""
    import lightgbm as lgb

    if seed is None:
        seed = RANDOM_SEED

    oof = np.zeros(n_train)
    test_preds = np.zeros(n_test)
    scores = []
    importances = np.zeros(len(feature_names))

    for fold, (tr_idx, va_idx) in enumerate(cv_splits):
        print(f"\n  Fold {fold} ...")
        params = LGBM_PARAMS.copy()
        params["random_state"] = seed
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train[tr_idx], y_train[tr_idx],
            eval_set=[(X_train[va_idx], y_train[va_idx])],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=200),
            ],
        )

        oof[va_idx] = model.predict(X_train[va_idx])
        test_preds += model.predict(X_test) / n_folds
        fold_r2 = r2_score(y_train[va_idx], oof[va_idx])
        scores.append(fold_r2)
        importances += model.feature_importances_ / n_folds
        print(f"  Fold {fold} R2: {fold_r2:.6f}")

        joblib.dump(model, os.path.join(MODEL_DIR, f"lgbm_fold{fold}_s{seed}.pkl"))

    overall = r2_score(y_train, oof)
    print(f"\n  LightGBM OOF R2: {overall:.6f}  (folds: {np.mean(scores):.6f} +/- {np.std(scores):.6f})")

    _save_importance(feature_names, importances, "lgbm")
    return oof, test_preds, scores


# =====================================================================
#  XGBoost
# =====================================================================
def _train_xgb(X_train, y_train, X_test, cv_splits, feature_names,
               n_train, n_test, n_folds):
    """Train XGBoost with OOF and return (oof, test_avg, scores)."""
    from xgboost import XGBRegressor

    oof = np.zeros(n_train)
    test_preds = np.zeros(n_test)
    scores = []
    importances = np.zeros(len(feature_names))

    for fold, (tr_idx, va_idx) in enumerate(cv_splits):
        print(f"\n  Fold {fold} ...")
        params = XGB_PARAMS.copy()
        params["early_stopping_rounds"] = 50
        model = XGBRegressor(**params)
        model.fit(
            X_train[tr_idx], y_train[tr_idx],
            eval_set=[(X_train[va_idx], y_train[va_idx])],
            verbose=200,
        )

        oof[va_idx] = model.predict(X_train[va_idx])
        test_preds += model.predict(X_test) / n_folds
        fold_r2 = r2_score(y_train[va_idx], oof[va_idx])
        scores.append(fold_r2)
        importances += model.feature_importances_ / n_folds
        print(f"  Fold {fold} R2: {fold_r2:.6f}")

        joblib.dump(model, os.path.join(MODEL_DIR, f"xgb_fold{fold}.pkl"))

    overall = r2_score(y_train, oof)
    print(f"\n  XGBoost OOF R2: {overall:.6f}  (folds: {np.mean(scores):.6f} +/- {np.std(scores):.6f})")

    _save_importance(feature_names, importances, "xgb")
    return oof, test_preds, scores


# =====================================================================
#  Feature importance helper
# =====================================================================
def _save_importance(feature_names, importances, model_name):
    """Save feature importance bar chart as PNG."""
    idx = np.argsort(importances)[::-1][:30]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [feature_names[i] for i in idx][::-1],
        importances[idx][::-1],
        color="#7209b7",
    )
    ax.set_title(f"Feature Importance -- {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"feature_importance_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -- Saved {path}")
