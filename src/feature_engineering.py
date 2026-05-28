"""
feature_engineering.py — Feature engineering for traffic demand prediction.

Creates temporal, cyclical, geohash, interaction, aggregate, and
target-encoded features.  All transformations are train-safe: encoders
and statistics are fitted on the training set and applied to the test set.

KEY INSIGHT: Train has day 48 (full) + day 49 (0:00-2:00).
             Test  has day 49 (2:15-23:45).
             So day-48 demand patterns per geohash are our strongest signal.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from config import (
    PEAK_HOURS, NIGHT_HOURS, TARGET_COL,
    N_FOLDS, RANDOM_SEED, TARGET_ENCODE_COLS,
)

warnings.filterwarnings("ignore")


# =====================================================================
#  HELPER: safe geohash decoding
# =====================================================================
def _decode_geohash(gh_series: pd.Series) -> pd.DataFrame:
    """Decode a Series of geohash strings to lat/lon.

    Parameters
    ----------
    gh_series : pd.Series
        Geohash strings.

    Returns
    -------
    pd.DataFrame
        Columns ``lat`` and ``lon``.
    """
    try:
        import pygeohash as pgh
        lats, lons = [], []
        for gh in gh_series:
            try:
                lat, lon = pgh.decode(str(gh))
                lats.append(float(lat))
                lons.append(float(lon))
            except Exception:
                lats.append(np.nan)
                lons.append(np.nan)
        return pd.DataFrame({"lat": lats, "lon": lons}, index=gh_series.index)
    except ImportError:
        print("WARNING: pygeohash not installed -- skipping geohash lat/lon decode")
        return pd.DataFrame(
            {"lat": np.nan, "lon": np.nan}, index=gh_series.index
        )


# =====================================================================
#  HELPER: K-Fold target encoding (leakage-safe) with smoothing
# =====================================================================
def kfold_target_encode(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list,
    target: str,
    n_folds: int = 5,
    smoothing: int = 10,
) -> tuple:
    """Leakage-safe KFold out-of-fold target encoding with smoothing.

    For each categorical column, the training set is encoded using OOF
    means (each fold's validation rows are encoded with means computed
    on the remaining folds).  The test set is encoded using the global
    mean computed on the full training set.  Smoothing prevents
    overfitting for categories with few samples.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe containing *target* column.
    test_df : pd.DataFrame
        Test dataframe (no target column expected).
    cols : list[str]
        Columns to target-encode.
    target : str
        Name of the target column.
    n_folds : int
        Number of folds for OOF encoding.
    smoothing : int
        Smoothing factor. Higher = more regularization toward global mean.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Augmented train_df and test_df with ``<col>_te`` columns added.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    global_mean = train_df[target].mean()

    for col in cols:
        if col not in train_df.columns:
            continue

        te_col = f"{col}_te"
        train_df[te_col] = np.nan

        for tr_idx, val_idx in kf.split(train_df):
            tr_fold = train_df.iloc[tr_idx]
            agg = tr_fold.groupby(col)[target].agg(["mean", "count"])
            # Smoothed mean: blend category mean with global mean based on count
            smooth_mean = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
            train_df.iloc[val_idx, train_df.columns.get_loc(te_col)] = (
                train_df.iloc[val_idx][col].map(smooth_mean)
            )

        # Fill any remaining NaNs with global mean
        train_df[te_col] = train_df[te_col].fillna(global_mean)

        # Test: full-train smoothed mean per category
        agg_full = train_df.groupby(col)[target].agg(["mean", "count"])
        smooth_full = (agg_full["count"] * agg_full["mean"] + smoothing * global_mean) / (agg_full["count"] + smoothing)
        test_df[te_col] = test_df[col].map(smooth_full).fillna(global_mean)

    return train_df, test_df


# =====================================================================
#  HELPER: Compute day-48 geohash demand profiles
# =====================================================================
def _compute_geohash_day48_profiles(train_df: pd.DataFrame) -> dict:
    """Compute per-geohash demand statistics from training data.

    Uses day 48 (full day) for hourly/time-slot patterns, and
    day 49 early morning for recency signal.

    Parameters
    ----------
    train_df : pd.DataFrame
        Full training dataframe with demand column.

    Returns
    -------
    dict
        Mapping dicts for each aggregate feature.
    """
    day48 = train_df[train_df["day"] == 48].copy() if "day" in train_df.columns else train_df.copy()

    parts = day48["timestamp"].astype(str).str.split(":", expand=True)
    day48["_hour"] = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype(int)
    day48["_minute"] = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype(int)
    day48["_time_slot"] = day48["_hour"] * 4 + day48["_minute"] // 15

    stats = {}

    # Per-geohash overall demand stats from day 48
    gh_agg = day48.groupby("geohash")[TARGET_COL].agg(["mean", "std", "median", "min", "max", "count"])
    stats["gh_demand_mean"] = gh_agg["mean"].to_dict()
    stats["gh_demand_std"] = gh_agg["std"].fillna(0).to_dict()
    stats["gh_demand_median"] = gh_agg["median"].to_dict()
    stats["gh_demand_min"] = gh_agg["min"].to_dict()
    stats["gh_demand_max"] = gh_agg["max"].to_dict()
    stats["gh_demand_count"] = gh_agg["count"].to_dict()

    # Per-geohash per-hour demand mean from day 48
    gh_hour_mean = day48.groupby(["geohash", "_hour"])[TARGET_COL].mean()
    stats["gh_hour_demand_mean"] = gh_hour_mean.to_dict()

    # Per-geohash per-TIME-SLOT demand from day 48 (4x finer than hour)
    gh_ts_mean = day48.groupby(["geohash", "_time_slot"])[TARGET_COL].mean()
    stats["gh_timeslot_demand_mean"] = gh_ts_mean.to_dict()

    # Per-geohash demand range (max - min) = volatility
    stats["gh_demand_range"] = (gh_agg["max"] - gh_agg["min"]).to_dict()

    # Per-geohash peak/off-peak
    peak_mask = day48["_hour"].isin(PEAK_HOURS)
    gh_peak = day48[peak_mask].groupby("geohash")[TARGET_COL].mean()
    gh_offpeak = day48[~peak_mask].groupby("geohash")[TARGET_COL].mean()
    stats["gh_peak_demand"] = gh_peak.to_dict()
    stats["gh_offpeak_demand"] = gh_offpeak.to_dict()

    # Day 49 early morning profiles (recency signal)
    if "day" in train_df.columns:
        day49 = train_df[train_df["day"] == 49].copy()
        if len(day49) > 0:
            gh_d49_mean = day49.groupby("geohash")[TARGET_COL].mean()
            stats["gh_demand_mean_d49"] = gh_d49_mean.to_dict()
            # Shift = how much did demand change from d48 to d49 early AM
            shift = gh_d49_mean - gh_agg["mean"].reindex(gh_d49_mean.index)
            stats["gh_demand_shift_d49"] = shift.fillna(0).to_dict()

    # Adjacent time-slot demand mapping from day 48
    # For each (geohash, time_slot), store demand at t-1, t+1, t-2, t+2
    ts_map = stats["gh_timeslot_demand_mean"]  # (geohash, slot) -> demand
    adj_maps = {}
    for offset in [-2, -1, 1, 2]:
        adj_map = {}
        for (gh, ts), val in ts_map.items():
            adj_ts = ts + offset
            if 0 <= adj_ts <= 95:
                adj_val = ts_map.get((gh, adj_ts), None)
                if adj_val is not None:
                    adj_map[(gh, ts)] = adj_val
        adj_maps[offset] = adj_map
    stats["gh_ts_adj"] = adj_maps

    # Prefix-level time-slot profiles (neighborhood temporal patterns)
    p5_ts = day48.copy()
    p5_ts["_prefix5"] = p5_ts["geohash"].str[:5]
    p5_ts_mean = p5_ts.groupby(["_prefix5", "_time_slot"])[TARGET_COL].mean()
    stats["p5_timeslot_demand_mean"] = p5_ts_mean.to_dict()

    # Global mean for fallback
    stats["global_mean"] = train_df[TARGET_COL].mean()

    return stats


# =====================================================================
#  MAIN: engineer_features
# =====================================================================
def engineer_features(
    df: pd.DataFrame,
    is_train: bool = True,
    train_stats: dict | None = None,
) -> tuple:
    """Create all engineered features for the traffic demand dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe (train or test).
    is_train : bool
        If True, computes and returns training statistics needed for
        the test set (geohash freq map, demand profiles, etc.).
    train_stats : dict or None
        Pre-computed training statistics (required when ``is_train=False``).

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (transformed df, stats dict)
    """
    if train_stats is None:
        train_stats = {}

    df = df.copy()

    # ── A. Timestamp features ────────────────────────────────────────
    if "timestamp" in df.columns:
        parts = df["timestamp"].astype(str).str.split(":", expand=True)
        df["hour"] = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype(int)
        df["minute"] = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype(int)
    else:
        df["hour"] = 0
        df["minute"] = 0

    if "day" in df.columns:
        df["weekday"] = df["day"].astype(int) % 7
        df["month"] = (df["day"].astype(int) // 30) % 12 + 1
        df["quarter"] = ((df["month"] - 1) // 3) + 1
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)
        df["weekofyear"] = df["day"].astype(int) // 7
    else:
        for c in ["weekday", "month", "quarter", "is_weekend", "weekofyear"]:
            df[c] = 0

    df["time_slot"] = df["hour"] * 4 + df["minute"] // 15
    df["peak_hour"] = df["hour"].isin(PEAK_HOURS).astype(int)
    df["night_hour"] = df["hour"].isin(NIGHT_HOURS).astype(int)

    # Time-of-day as fraction (0.0 to 1.0)
    df["time_frac"] = (df["hour"] * 60 + df["minute"]) / (24 * 60)

    # ── B. Cyclical encoding ─────────────────────────────────────────
    for col, period in [("hour", 24), ("minute", 60), ("weekday", 7), ("time_slot", 96)]:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

    # ── C. Geohash features ──────────────────────────────────────────
    if "geohash" in df.columns:
        df["geohash"] = df["geohash"].astype(str)
        geo_decoded = _decode_geohash(df["geohash"])
        df["lat"] = geo_decoded["lat"]
        df["lon"] = geo_decoded["lon"]
        df["geohash_prefix_4"] = df["geohash"].str[:4]
        df["geohash_prefix_5"] = df["geohash"].str[:5]

        # Frequency encoding
        if is_train:
            freq_map = df["geohash"].value_counts().to_dict()
            train_stats["geohash_freq_map"] = freq_map
        else:
            freq_map = train_stats.get("geohash_freq_map", {})
        df["geohash_freq"] = df["geohash"].map(freq_map).fillna(0).astype(int)

    # ── D. Categorical encoding (binary) ─────────────────────────────
    if "LargeVehicles" in df.columns:
        df["large_vehicles_flag"] = (df["LargeVehicles"] == "Allowed").astype(int)
    if "Landmarks" in df.columns:
        df["landmarks_flag"] = (df["Landmarks"] == "Yes").astype(int)

    # ── E. Interaction features ──────────────────────────────────────
    df["geohash_time_slot"] = (
        df["geohash"].astype(str) + "_" + df["time_slot"].astype(str)
    )
    df["geohash_hour"] = (
        df["geohash"].astype(str) + "_" + df["hour"].astype(str)
    )
    if "Weather" in df.columns:
        df["weather_hour"] = df["Weather"].astype(str) + "_" + df["hour"].astype(str)
    else:
        df["weather_hour"] = "UNK_0"
    if "RoadType" in df.columns:
        df["roadtype_hour"] = df["RoadType"].astype(str) + "_" + df["hour"].astype(str)
        df["roadtype_lanes"] = df["RoadType"].astype(str) + "_" + df["NumberofLanes"].astype(str)
    else:
        df["roadtype_hour"] = "UNK_0"
        df["roadtype_lanes"] = "UNK_0"
    df["weekday_hour"] = df["weekday"].astype(str) + "_" + df["hour"].astype(str)

    # Temperature binning
    if "Temperature" in df.columns:
        df["temp_bin"] = pd.cut(
            df["Temperature"], bins=10, labels=False
        )
        df["temp_bin"] = df["temp_bin"].fillna(-1).astype(int)
    else:
        df["temp_bin"] = -1
    df["temp_hour"] = df["temp_bin"].astype(str) + "_" + df["hour"].astype(str)

    # ── F. Day-48 geohash demand profiles (KEY FEATURES) ─────────────
    if is_train:
        profiles = _compute_geohash_day48_profiles(df)
        train_stats["profiles"] = profiles
    else:
        profiles = train_stats.get("profiles", {})

    global_mean = profiles.get("global_mean", 0.0)

    # Map geohash-level aggregate features
    df["gh_demand_mean_d48"] = df["geohash"].map(profiles.get("gh_demand_mean", {})).fillna(global_mean)
    df["gh_demand_std_d48"] = df["geohash"].map(profiles.get("gh_demand_std", {})).fillna(0)
    df["gh_demand_median_d48"] = df["geohash"].map(profiles.get("gh_demand_median", {})).fillna(global_mean)
    df["gh_demand_min_d48"] = df["geohash"].map(profiles.get("gh_demand_min", {})).fillna(0)
    df["gh_demand_max_d48"] = df["geohash"].map(profiles.get("gh_demand_max", {})).fillna(global_mean)
    df["gh_demand_range_d48"] = df["geohash"].map(profiles.get("gh_demand_range", {})).fillna(0)
    df["gh_demand_count_d48"] = df["geohash"].map(profiles.get("gh_demand_count", {})).fillna(0)
    df["gh_peak_demand_d48"] = df["geohash"].map(profiles.get("gh_peak_demand", {})).fillna(global_mean)
    df["gh_offpeak_demand_d48"] = df["geohash"].map(profiles.get("gh_offpeak_demand", {})).fillna(global_mean)

    # Per-geohash per-hour demand from day 48
    gh_hour_map = profiles.get("gh_hour_demand_mean", {})
    df["gh_hour_demand_d48"] = df.apply(
        lambda row: gh_hour_map.get((row["geohash"], row["hour"]), global_mean), axis=1
    )

    # Per-geohash per-TIME-SLOT demand from day 48 (15-min granularity)
    gh_ts_map = profiles.get("gh_timeslot_demand_mean", {})
    df["gh_timeslot_demand_d48"] = df.apply(
        lambda row: gh_ts_map.get((row["geohash"], row["time_slot"]), global_mean), axis=1
    )

    # Adjacent time-slot demand (temporal gradient from day 48)
    adj_maps = profiles.get("gh_ts_adj", {})
    for offset, label in [(-1, "prev1"), (1, "next1"), (-2, "prev2"), (2, "next2")]:
        adj_map = adj_maps.get(offset, {})
        df[f"gh_ts_{label}_d48"] = df.apply(
            lambda row, am=adj_map: am.get((row["geohash"], row["time_slot"]), global_mean), axis=1
        )

    # Temporal gradient features
    df["gh_ts_gradient_d48"] = df["gh_ts_next1_d48"] - df["gh_ts_prev1_d48"]

    # ── G. Cyclical encoding for interaction features ──────────────────
    df["time_slot"] = df["hour"] * 4 + df["minute"] // 15
    df["peak_hour"] = df["hour"].isin(PEAK_HOURS).astype(int)
    df["night_hour"] = df["hour"].isin(NIGHT_HOURS).astype(int)

    # Day 49 early morning signal (recency)
    gh_d49_map = profiles.get("gh_demand_mean_d49", {})
    df["gh_demand_mean_d49"] = df["geohash"].map(gh_d49_map).fillna(global_mean)
    gh_shift_map = profiles.get("gh_demand_shift_d49", {})
    df["gh_demand_shift_d49"] = df["geohash"].map(gh_shift_map).fillna(0)

    # Calibrated demand: scale day48 hourly profile by global calibration ratio
    # Simple approach: use overall day49 early / day48 early ratio
    df["gh_hour_calibrated"] = df["gh_hour_demand_d48"].copy()
    df["gh_timeslot_calibrated"] = df["gh_timeslot_demand_d48"].copy()

    # Prefix-5 time-slot demand (neighborhood temporal pattern)
    p5_ts_map = profiles.get("p5_timeslot_demand_mean", {})
    df["p5_timeslot_demand_d48"] = df.apply(
        lambda row: p5_ts_map.get((row["geohash"][:5], row["time_slot"]), global_mean), axis=1
    )

    # Ratio features
    df["demand_vs_mean_ratio"] = df["gh_hour_demand_d48"] / (df["gh_demand_mean_d48"] + 1e-8)
    df["demand_vs_peak_ratio"] = df["gh_hour_demand_d48"] / (df["gh_peak_demand_d48"] + 1e-8)

    # ── G. Prefix-level aggregates ───────────────────────────────────
    if is_train:
        p4_mean = df.groupby("geohash_prefix_4")[TARGET_COL].mean().to_dict()
        p5_mean = df.groupby("geohash_prefix_5")[TARGET_COL].mean().to_dict()
        train_stats["p4_mean"] = p4_mean
        train_stats["p5_mean"] = p5_mean
    else:
        p4_mean = train_stats.get("p4_mean", {})
        p5_mean = train_stats.get("p5_mean", {})

    df["prefix4_demand_mean"] = df["geohash_prefix_4"].map(p4_mean).fillna(global_mean)
    df["prefix5_demand_mean"] = df["geohash_prefix_5"].map(p5_mean).fillna(global_mean)

    print(f"[OK] Feature engineering {'(train)' if is_train else '(test)'} done -- "
          f"{df.shape[1]} columns")
    return df, train_stats


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the list of feature columns (excluding target, Index, raw strings).

    Parameters
    ----------
    df : pd.DataFrame
        Engineered dataframe.

    Returns
    -------
    list[str]
        Feature column names suitable for model training.
    """
    drop_cols = {
        TARGET_COL, "Index", "timestamp", "geohash",
        "geohash_prefix_4", "geohash_prefix_5",
        "geohash_time_slot", "geohash_hour",
        "weather_hour", "roadtype_hour", "roadtype_lanes",
        "weekday_hour", "temp_hour", "day",
        "LargeVehicles", "Landmarks", "Weather", "RoadType",
    }
    # Also drop any remaining object columns (safety net)
    obj_cols = set(df.select_dtypes(include=["object"]).columns)
    exclude = drop_cols | obj_cols

    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols
