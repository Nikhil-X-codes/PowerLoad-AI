"""Phase 4: Exploratory Data Analysis on development data (Jan-Nov) only."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "outputs" / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.load_prediction.data import (
    EXPECTED_CLASSES,
    NUMERIC_COLUMNS,
    chronological_split,
    clean_dataset,
    load_raw_data,
)
from src.load_prediction.features import derive_features

sns.set_theme(style="whitegrid", context="notebook")


def generate_eda_figures(
    dev_df: pd.DataFrame,
    raw_dev_df: pd.DataFrame,
    figures_dir: str | Path,
) -> dict[str, Any]:
    """Generate all Phase 4 EDA plots using development data (Jan-Nov) only."""
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_df = derive_features(dev_df)
    feat_df["Load_Type"] = dev_df["Load_Type"].values

    # 1. Target class distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = feat_df["Load_Type"].value_counts().reindex(EXPECTED_CLASSES)
    total = len(feat_df)
    bars = ax.bar(counts.index, counts.values, color=["#4c72b0", "#dd8452", "#c44e52"])
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + total * 0.01,
            f"{val:,}\n({val / total * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Development Set (Jan-Nov) Target Class Distribution")
    ax.set_ylabel("Row Count")
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()
    fig.savefig(out_dir / "01_target_class_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Hourly and monthly electricity consumption
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    hourly = feat_df.groupby("hour")["Usage_kWh"].agg(["mean", "median"])
    axes[0].plot(hourly.index, hourly["mean"], marker="o", label="Mean Usage_kWh")
    axes[0].plot(hourly.index, hourly["median"], marker="s", linestyle="--", label="Median Usage_kWh")
    axes[0].set_title("Hourly Electricity Consumption (Dev Set)")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Usage_kWh")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].legend()

    monthly = feat_df.groupby("month")["Usage_kWh"].agg(["mean", "median"])
    axes[1].plot(monthly.index, monthly["mean"], marker="o", label="Mean Usage_kWh")
    axes[1].plot(monthly.index, monthly["median"], marker="s", linestyle="--", label="Median Usage_kWh")
    axes[1].set_title("Monthly Electricity Consumption (Jan-Nov)")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Usage_kWh")
    axes[1].set_xticks(range(1, 12))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "02_hourly_monthly_usage.png", dpi=150)
    plt.close(fig)

    # 3. Missing-value distribution (raw vs cleaned on dev set)
    raw_missing = raw_dev_df[list(NUMERIC_COLUMNS)].isna().sum()
    clean_missing = dev_df[list(NUMERIC_COLUMNS)].isna().sum()
    miss_df = pd.DataFrame({"Raw": raw_missing, "After Cleaning (Invalid->NaN)": clean_missing})
    fig, ax = plt.subplots(figsize=(9, 4.8))
    miss_df.plot(kind="barh", ax=ax, color=["#4c72b0", "#dd8452"])
    ax.set_title("Missing Values in Development Set (Raw vs Post-Cleaning)")
    ax.set_xlabel("Missing / Invalid Count")
    fig.tight_layout()
    fig.savefig(out_dir / "03_missing_value_distribution.png", dpi=150)
    plt.close(fig)

    # 4. Feature correlation heatmap
    corr_cols = list(NUMERIC_COLUMNS) + ["hour", "day_of_week", "is_weekend", "month", "total_reactive_power_magnitude"]
    corr = feat_df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Heatmap (Development Set)")
    fig.tight_layout()
    fig.savefig(out_dir / "04_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # 5. Power consumption by load category
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.boxplot(
        data=feat_df,
        x="Load_Type",
        y="Usage_kWh",
        order=EXPECTED_CLASSES,
        palette=["#4c72b0", "#dd8452", "#c44e52"],
        ax=ax,
    )
    ax.set_title("Electricity Consumption (Usage_kWh) by Load Category (Dev Set)")
    ax.set_ylabel("Usage_kWh")
    fig.tight_layout()
    fig.savefig(out_dir / "05_usage_by_load_type.png", dpi=150)
    plt.close(fig)

    # 6. Feature distributions and outlier box plots
    box_cols = [
        "Usage_kWh",
        "Lagging_Current_Reactive.Power_kVarh",
        "Leading_Current_Reactive_Power_kVarh",
        "CO2(tCO2)",
        "Lagging_Current_Power_Factor",
        "Leading_Current_Power_Factor",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), box_cols):
        sns.boxplot(y=feat_df[col].dropna(), ax=ax, color="#4c72b0")
        ax.set_title(col, fontsize=9)
    fig.suptitle("Numeric Feature Distributions & Outliers (Development Set)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "06_feature_distributions_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 7. Weekday vs weekend consumption patterns
    fig, ax = plt.subplots(figsize=(9, 4.8))
    wk = feat_df.groupby(["is_weekend", "hour"])["Usage_kWh"].mean().unstack(0)
    ax.plot(wk.index, wk[0.0], marker="o", label="Weekday (Mon-Fri)")
    ax.plot(wk.index, wk[1.0], marker="s", linestyle="--", label="Weekend (Sat-Sun)")
    ax.set_title("Average Hourly Usage_kWh: Weekday vs Weekend (Dev Set)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Usage_kWh")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "07_weekday_vs_weekend.png", dpi=150)
    plt.close(fig)

    usage_stats = feat_df.groupby("Load_Type")["Usage_kWh"].agg(["count", "mean", "median", "std"]).to_dict("index")
    return {
        "dev_rows": int(len(feat_df)),
        "class_counts": counts.to_dict(),
        "usage_by_load_type": {
            k: {m: round(float(v), 3) for m, v in stats.items()}
            for k, stats in usage_stats.items()
        },
        "figures_generated": sorted(p.name for p in out_dir.glob("*.png")),
    }


def main() -> None:
    """Run Phase 4 EDA figure generation on Jan-Nov development data."""
    project_root = Path(__file__).resolve().parents[2]
    raw_df = load_raw_data(project_root / "data" / "load_data.csv")
    cleaned_df, _ = clean_dataset(raw_df)
    train_df, val_df, _, split_summary = chronological_split(cleaned_df)

    # Use raw Jan-Nov subset only for raw vs cleaned missingness comparison
    raw_sorted = raw_df.sort_values("Date_Time", kind="mergesort").reset_index(drop=True)
    raw_dev = raw_sorted.loc[raw_sorted["Date_Time"] <= val_df["Date_Time"].max()]
    dev_df = pd.concat([train_df, val_df], ignore_index=True)

    eda_summary = generate_eda_figures(dev_df, raw_dev, project_root / "outputs" / "figures")
    print(json.dumps({"split_summary": split_summary, "eda_summary": eda_summary}, indent=2))


if __name__ == "__main__":
    main()
