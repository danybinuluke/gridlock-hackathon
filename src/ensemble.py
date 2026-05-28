"""
ensemble.py — Optimal blending + stacking of OOF predictions.

Uses ``scipy.optimize.minimize`` for optimal blending weights,
plus a Ridge meta-learner for stacking. The best method is used.
Final predictions are clipped to [0, 1].
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from config import RANDOM_SEED, N_FOLDS


def optimize_ensemble(oof_dict: dict, y_train: np.ndarray) -> dict:
    """Find optimal ensemble weights and stacking, return the best.

    Parameters
    ----------
    oof_dict : dict
        ``{model_name: oof_predictions}`` — OOF arrays of shape ``(n_train,)``.
    y_train : np.ndarray
        True target values.

    Returns
    -------
    dict
        Contains 'method', 'weights' or 'meta_model', 'r2'.
    """
    model_names = list(oof_dict.keys())
    oof_matrix = np.column_stack([oof_dict[m] for m in model_names])
    n_models = len(model_names)

    # ── Method 1: Weighted blend ─────────────────────────────────────
    def _neg_r2(weights):
        blended = oof_matrix @ weights
        return -r2_score(y_train, blended)

    x0 = np.ones(n_models) / n_models
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    bounds = [(0.0, 1.0)] * n_models

    result = minimize(
        _neg_r2, x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    blend_weights = {m: round(float(w), 6) for m, w in zip(model_names, result.x)}
    blend_r2 = -result.fun

    print("\n" + "=" * 60)
    print("ENSEMBLE — Optimal Weights (Blend)")
    print("=" * 60)
    for m, w in blend_weights.items():
        print(f"  {m:>12s}: {w:.6f}")
    print(f"\n  Blend OOF R2: {blend_r2:.6f}")

    # ── Method 2: Ridge stacking ─────────────────────────────────────
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    stack_oof = np.zeros(len(y_train))

    for tr_idx, va_idx in kf.split(oof_matrix):
        meta = Ridge(alpha=1.0)
        meta.fit(oof_matrix[tr_idx], y_train[tr_idx])
        stack_oof[va_idx] = meta.predict(oof_matrix[va_idx])

    stack_r2 = r2_score(y_train, stack_oof)

    # Fit final meta model on all data
    meta_final = Ridge(alpha=1.0)
    meta_final.fit(oof_matrix, y_train)

    print(f"\n  Stacking OOF R2: {stack_r2:.6f}")

    # Pick best method
    if stack_r2 > blend_r2:
        print(f"\n  >> Using STACKING (R2 {stack_r2:.6f} > {blend_r2:.6f})")
        return {
            "method": "stacking",
            "meta_model": meta_final,
            "model_names": model_names,
            "r2": stack_r2,
        }
    else:
        print(f"\n  >> Using BLEND (R2 {blend_r2:.6f} >= {stack_r2:.6f})")
        return {
            "method": "blend",
            "weights": blend_weights,
            "model_names": model_names,
            "r2": blend_r2,
        }


def blend_predictions(test_preds_dict: dict, ensemble_info: dict) -> np.ndarray:
    """Apply ensemble to test predictions and clip to [0, 1].

    Parameters
    ----------
    test_preds_dict : dict
        ``{model_name: test_predictions_array}``.
    ensemble_info : dict
        Output from ``optimize_ensemble``.

    Returns
    -------
    np.ndarray
        Final clipped predictions.
    """
    model_names = ensemble_info["model_names"]

    if ensemble_info["method"] == "stacking":
        test_matrix = np.column_stack([test_preds_dict[m] for m in model_names])
        preds = ensemble_info["meta_model"].predict(test_matrix)
    else:
        weights = ensemble_info["weights"]
        preds = np.zeros_like(test_preds_dict[model_names[0]])
        for m in model_names:
            preds += weights[m] * test_preds_dict[m]

    # Clip to valid range [0, 1]
    preds = np.clip(preds, 0.0, 1.0)
    return preds
