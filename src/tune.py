"""
tune.py — Optuna-based hyperparameter tuning for LightGBM.

Provides ``tune_lightgbm`` which runs a Bayesian search over the
LightGBM hyperparameter space, maximising mean OOF R² across the
provided time-series CV splits.
"""

import warnings
import numpy as np
from sklearn.metrics import r2_score

from config import RANDOM_SEED

warnings.filterwarnings("ignore")


def tune_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_splits: list,
    n_trials: int = 50,
) -> dict:
    """Tune LightGBM hyperparameters using Optuna.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target vector.
    cv_splits : list[tuple]
        ``(train_idx, val_idx)`` from ``get_cv_splits``.
    n_trials : int
        Number of Optuna trials.

    Returns
    -------
    dict
        Best hyperparameters found.
    """
    import optuna
    import lightgbm as lgb

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        """Optuna objective: mean OOF R² over CV splits."""
        params = {
            "n_estimators": 3000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "verbose": -1,
        }

        fold_scores = []
        for tr_idx, va_idx in cv_splits:
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train[tr_idx], y_train[tr_idx],
                eval_set=[(X_train[va_idx], y_train[va_idx])],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                ],
            )
            preds = model.predict(X_train[va_idx])
            fold_scores.append(r2_score(y_train[va_idx], preds))

        return np.mean(fold_scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print("\n" + "=" * 60)
    print("OPTUNA — Best LightGBM Params")
    print("=" * 60)
    for k, v in best.items():
        print(f"  {k}: {v}")
    print(f"\n  Best mean OOF R²: {study.best_value:.6f}")

    return best
