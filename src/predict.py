"""
predict.py — Main runner for the traffic demand prediction pipeline.

Orchestrates EDA → feature engineering → preprocessing → validation →
model training → ensemble → submission generation → SHAP explainability.

Usage
-----
    python src/predict.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# Ensure src/ is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TRAIN_FILE, TEST_FILE, OUTPUT_DIR, MODEL_DIR,
    TARGET_COL, INDEX_COL, N_FOLDS, TARGET_ENCODE_COLS,
)
from eda import run_eda
from feature_engineering import (
    engineer_features, get_feature_columns, kfold_target_encode,
)
from preprocessing import preprocess
from validation import get_cv_splits
from models import train_models
from ensemble import optimize_ensemble, blend_predictions

warnings.filterwarnings("ignore")


def main():
    """Run the full traffic demand prediction pipeline end-to-end."""
    print("Starting pipeline...\n")

    # ─────────────────────────────────────────────────────────────────
    # 1. Load data
    # ─────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 — Loading data")
    print("=" * 60)
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    # Save test Index for final submission
    test_index = test_df[INDEX_COL].values.copy()

    # ─────────────────────────────────────────────────────────────────
    # 2. EDA
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — EDA")
    print("=" * 60)
    run_eda(train_df)

    # ─────────────────────────────────────────────────────────────────
    # 3. Feature engineering
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — Feature Engineering")
    print("=" * 60)
    train_df, train_stats = engineer_features(train_df, is_train=True)
    test_df, _ = engineer_features(test_df, is_train=False, train_stats=train_stats)

    # Target encoding (leakage-safe)
    te_cols = [c for c in TARGET_ENCODE_COLS if c in train_df.columns]
    if te_cols:
        train_df, test_df = kfold_target_encode(
            train_df, test_df, te_cols, TARGET_COL, n_folds=N_FOLDS
        )
        print(f"✓ Target-encoded columns: {te_cols}")

    # ─────────────────────────────────────────────────────────────────
    # 4. Preprocessing
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — Preprocessing")
    print("=" * 60)
    feature_cols = get_feature_columns(train_df)
    X_train, y_train, X_test, final_features = preprocess(
        train_df, test_df.copy(), feature_cols
    )

    # ─────────────────────────────────────────────────────────────────
    # 5. CV splits
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 — Validation splits")
    print("=" * 60)
    timestamps = train_df["day"].values if "day" in train_df.columns else np.arange(len(X_train))
    cv_splits = get_cv_splits(X_train, y_train, timestamps, n_folds=N_FOLDS)

    # ─────────────────────────────────────────────────────────────────
    # 6. Model training
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6 — Training models")
    print("=" * 60)
    results = train_models(X_train, y_train, X_test, cv_splits, final_features)

    # ─────────────────────────────────────────────────────────────────
    # 7. Ensemble
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7 — Ensemble optimisation")
    print("=" * 60)
    oof_dict = {name: vals[0] for name, vals in results.items()}
    test_dict = {name: vals[1] for name, vals in results.items()}

    ensemble_info = optimize_ensemble(oof_dict, y_train)
    final_preds = blend_predictions(test_dict, ensemble_info)

    # ─────────────────────────────────────────────────────────────────
    # 8. Save submission
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8 — Saving outputs")
    print("=" * 60)

    # Submission CSV
    submission = pd.DataFrame({
        INDEX_COL: test_index,
        TARGET_COL: final_preds,
    })
    # Safety: fill any NaN with global mean
    if submission[TARGET_COL].isna().any():
        submission[TARGET_COL].fillna(y_train.mean(), inplace=True)
    sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"✓ submission.csv saved — {len(submission)} rows")
    print(f"  NaN count: {submission[TARGET_COL].isna().sum()}")
    assert len(submission) == len(test_index), (
        f"Expected {len(test_index)} rows, got {len(submission)}"
    )

    # Feature importance CSV (averaged across models)
    _save_feature_importance_csv(results, final_features)

    # Validation scores text
    _save_validation_scores(results, oof_dict, y_train, ensemble_info)

    # ─────────────────────────────────────────────────────────────────
    # 9. SHAP explainability
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 9 — SHAP explainability")
    print("=" * 60)
    _run_shap(X_train, final_features)

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)


# =====================================================================
#  SHAP
# =====================================================================
def _run_shap(X_train, feature_names):
    """Generate SHAP beeswarm plot for the first LightGBM fold model.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    feature_names : list[str]
        Feature column names.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import shap
        import joblib

        model_path = os.path.join(MODEL_DIR, "lgbm_fold0.pkl")
        if not os.path.exists(model_path):
            print("⚠ lgbm_fold0.pkl not found — skipping SHAP")
            return

        model = joblib.load(model_path)

        # Use a subsample for speed
        n_sample = min(2000, X_train.shape[0])
        idx = np.random.RandomState(42).choice(X_train.shape[0], n_sample, replace=False)
        X_sample = X_train[idx]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feature_names,
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "shap_summary.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ SHAP beeswarm plot saved to {path}")

        # Print top-10 features by mean |SHAP|
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:10]
        print("\n  Top 10 features by mean |SHAP|:")
        for i, ix in enumerate(top_idx, 1):
            print(f"    {i:2d}. {feature_names[ix]:30s}  {mean_abs[ix]:.6f}")

    except Exception as e:
        print(f"⚠ SHAP failed: {e}")


# =====================================================================
#  Feature importance CSV
# =====================================================================
def _save_feature_importance_csv(results, feature_names):
    """Save mean feature importance across all models and folds.

    Parameters
    ----------
    results : dict
        Model training results.
    feature_names : list[str]
        Feature names.
    """
    import joblib

    imp_dict = {"feature": feature_names}
    for model_name in results:
        try:
            m = joblib.load(os.path.join(MODEL_DIR, f"{model_name}_fold0.pkl"))
            if hasattr(m, "feature_importances_"):
                imp = m.feature_importances_
            elif hasattr(m, "get_feature_importance"):
                imp = m.get_feature_importance()
            else:
                imp = np.zeros(len(feature_names))

            # Pad or trim to match feature_names length
            if len(imp) != len(feature_names):
                imp_padded = np.zeros(len(feature_names))
                imp_padded[:len(imp)] = imp[:len(feature_names)]
                imp = imp_padded
            imp_dict[model_name] = imp
        except Exception as e:
            print(f"  ⚠ {model_name} importance failed: {e}")
            imp_dict[model_name] = np.zeros(len(feature_names))

    imp_df = pd.DataFrame(imp_dict)
    model_cols = [c for c in imp_df.columns if c != "feature"]
    imp_df["mean_importance"] = imp_df[model_cols].mean(axis=1)
    imp_df = imp_df.sort_values("mean_importance", ascending=False)
    path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    imp_df.to_csv(path, index=False)
    print(f"✓ feature_importance.csv saved — {len(imp_df)} features")


# =====================================================================
#  Validation scores text
# =====================================================================
def _save_validation_scores(results, oof_dict, y_train, ensemble_info):
    """Save per-model per-fold R2 and ensemble R2 to a text file.

    Parameters
    ----------
    results : dict
    oof_dict : dict
    y_train : np.ndarray
    ensemble_info : dict
    """
    lines = []
    lines.append("=" * 60)
    lines.append("VALIDATION SCORES")
    lines.append("=" * 60)

    for name, (oof, test, scores) in results.items():
        lines.append(f"\n{name.upper()}")
        for i, s in enumerate(scores):
            lines.append(f"  Fold {i}: R2 = {s:.6f}")
        overall = r2_score(y_train, oof)
        lines.append(f"  OOF R2:  {overall:.6f}")
        lines.append(f"  Mean +/- Std: {np.mean(scores):.6f} +/- {np.std(scores):.6f}")

    # Ensemble
    method = ensemble_info.get('method', 'blend')
    ens_r2 = ensemble_info.get('r2', 0)
    lines.append(f"\nENSEMBLE ({method.upper()})")
    if method == 'blend':
        lines.append(f"  Weights: {ensemble_info.get('weights', {})}")
    else:
        lines.append(f"  Meta-model: Ridge stacking")
    lines.append(f"  OOF R2:  {ens_r2:.6f}")
    lines.append("=" * 60)

    txt = "\n".join(lines)
    path = os.path.join(OUTPUT_DIR, "validation_scores.txt")
    with open(path, "w") as f:
        f.write(txt)
    print(f"-- validation_scores.txt saved")
    print(txt)


if __name__ == "__main__":
    main()
