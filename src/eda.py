"""
eda.py — Exploratory Data Analysis for the traffic demand dataset.

Generates descriptive statistics, distribution plots, and correlation
heatmaps. All plots are saved as PNG files to submissions/eda/.
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from config import EDA_DIR, TARGET_COL

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def run_eda(train_df: pd.DataFrame) -> None:
    """Run full exploratory data analysis on the training set.

    Parameters
    ----------
    train_df : pd.DataFrame
        Raw training dataframe with the ``demand`` target column.

    Side-effects
    -------------
    * Prints summary statistics to stdout.
    * Saves PNG plots to ``submissions/eda/``.
    """
    os.makedirs(EDA_DIR, exist_ok=True)

    # ── 1. Basic info ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EDA — BASIC INFO")
    print("=" * 60)
    print(f"Shape: {train_df.shape}")
    print(f"\nDtypes:\n{train_df.dtypes}")
    print(f"\nMissing values:\n{train_df.isnull().sum()}")
    print(f"\nMissing percentage:\n{(train_df.isnull().mean() * 100).round(2)}")
    print(f"\nUnique counts:\n{train_df.nunique()}")
    print(f"\nDescribe:\n{train_df.describe()}")

    # ── Parse timestamp once for EDA plots ───────────────────────────
    df = train_df.copy()
    if "timestamp" in df.columns:
        parts = df["timestamp"].astype(str).str.split(":", expand=True)
        df["_hour"] = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype(int)
    if "day" in df.columns:
        df["_weekday"] = df["day"].astype(int) % 7

    # ── 2. Demand distribution ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[TARGET_COL].dropna(), bins=80, kde=True, ax=ax, color="#5e60ce")
    ax.set_title("Demand Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("demand")
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "demand_distribution.png"), dpi=150)
    plt.close(fig)
    print("✓ Saved demand_distribution.png")

    # ── 3. Demand vs hour ────────────────────────────────────────────
    if "_hour" in df.columns:
        hourly = df.groupby("_hour")[TARGET_COL].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hourly["_hour"], hourly["mean"], marker="o", color="#6930c3")
        ax.fill_between(
            hourly["_hour"],
            hourly["mean"] - hourly["std"],
            hourly["mean"] + hourly["std"],
            alpha=0.2,
            color="#6930c3",
        )
        ax.set_title("Demand vs Hour (mean ± std)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Demand")
        fig.tight_layout()
        fig.savefig(os.path.join(EDA_DIR, "demand_vs_hour.png"), dpi=150)
        plt.close(fig)
        print("✓ Saved demand_vs_hour.png")

    # ── 4. Demand vs weekday ─────────────────────────────────────────
    if "_weekday" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=df, x="_weekday", y=TARGET_COL, ax=ax,
                    palette="viridis", errorbar="sd")
        ax.set_title("Demand vs Weekday", fontsize=14, fontweight="bold")
        ax.set_xlabel("Weekday (0=Mon approx)")
        ax.set_ylabel("Demand")
        fig.tight_layout()
        fig.savefig(os.path.join(EDA_DIR, "demand_vs_weekday.png"), dpi=150)
        plt.close(fig)
        print("✓ Saved demand_vs_weekday.png")

    # ── 5. Demand vs Weather (box plot) ──────────────────────────────
    if "Weather" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df, x="Weather", y=TARGET_COL, ax=ax,
                    palette="Set2")
        ax.set_title("Demand vs Weather", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(EDA_DIR, "demand_vs_weather.png"), dpi=150)
        plt.close(fig)
        print("✓ Saved demand_vs_weather.png")

    # ── 6. Outlier detection (IQR) ───────────────────────────────────
    q1 = df[TARGET_COL].quantile(0.25)
    q3 = df[TARGET_COL].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df[TARGET_COL] < lower) | (df[TARGET_COL] > upper)]
    pct = len(outliers) / len(df) * 100
    print(f"\nOutlier detection (IQR): {len(outliers)} rows "
          f"({pct:.2f}%) outside [{lower:.4f}, {upper:.4f}]")

    # ── 7. Correlation heatmap ───────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # drop helper cols
    numeric_cols = [c for c in numeric_cols if not c.startswith("_")]
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    square=True, ax=ax, linewidths=0.5)
        ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(EDA_DIR, "correlation_heatmap.png"), dpi=150)
        plt.close(fig)
        print("✓ Saved correlation_heatmap.png")

    print("\n✅ EDA complete — all plots saved to", EDA_DIR)
