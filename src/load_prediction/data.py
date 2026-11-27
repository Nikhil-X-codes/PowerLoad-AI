"""Data loading, schema validation, and Phase 1 data quality audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_COLUMNS: tuple[str, ...] = (
    "Date_Time",
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
    "Load_Type",
)

EXPECTED_CLASSES: tuple[str, ...] = (
    "Light_Load",
    "Medium_Load",
    "Maximum_Load",
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
)


def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    """Load raw load_data.csv, validate schema and target labels, and parse day-first timestamps."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    if missing_cols or extra_cols or tuple(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Schema mismatch. Expected {EXPECTED_COLUMNS}, got {tuple(df.columns)}. "
            f"Missing={missing_cols}, Extra={extra_cols}"
        )

    parsed_dt = pd.to_datetime(df["Date_Time"], dayfirst=True, errors="raise")
    if parsed_dt.isna().any():
        raise ValueError("Null timestamps encountered after day-first parsing.")
    df["Date_Time"] = parsed_dt

    observed_classes = set(df["Load_Type"].dropna().unique())
    if df["Load_Type"].isna().any():
        raise ValueError("Target column 'Load_Type' contains missing values.")
    unexpected_classes = observed_classes - set(EXPECTED_CLASSES)
    if unexpected_classes:
        raise ValueError(f"Unexpected target labels found: {sorted(unexpected_classes)}")

    return df


def audit_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Compute data quality metrics for the loaded dataset."""
    class_counts = {
        cls: int((df["Load_Type"] == cls).sum()) for cls in EXPECTED_CLASSES
    }
    missing_counts = {col: int(df[col].isna().sum()) for col in df.columns}

    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
        "date_range": {
            "min": df["Date_Time"].min().isoformat(),
            "max": df["Date_Time"].max().isoformat(),
        },
        "class_counts": class_counts,
        "missing_counts": missing_counts,
        "duplicate_timestamps": int(df["Date_Time"].duplicated().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "is_monotonic_increasing": bool(df["Date_Time"].is_monotonic_increasing),
    }


def compute_timestamp_nsm(dt_series: pd.Series) -> pd.Series:
    """Compute seconds since midnight (0..86399) deterministically from Date_Time."""
    return (
        dt_series.dt.hour * 3600
        + dt_series.dt.minute * 60
        + dt_series.dt.second
    ).astype(float)


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply deterministic Phase 2 cleaning without fitting any data-dependent statistics.

    - Preserves the input DataFrame (operates on a copy).
    - Sorts chronologically by Date_Time (stable mergesort).
    - Flags duplicate timestamps without dropping rows.
    - Marks physically invalid Power Factor values (< 0 or > 100) as NaN and records indicators.
    - Compares NSM against timestamp-derived seconds since midnight, flags mismatches,
      and marks out-of-range (< 0 or > 86400) or mismatched NSM values as NaN.
    - Preserves extreme but plausible Usage_kWh, Reactive Power, and CO2 measurements.
    """
    cleaned = df.copy()
    cleaned = cleaned.sort_values("Date_Time", kind="mergesort").reset_index(drop=True)

    duplicate_ts_count = int(cleaned["Date_Time"].duplicated().sum())

    # Power factor validity checks: expected percentage range [0, 100]
    pf_invalid_counts: dict[str, int] = {}
    for pf_col in ("Lagging_Current_Power_Factor", "Leading_Current_Power_Factor"):
        invalid_mask = cleaned[pf_col].notna() & (
            (cleaned[pf_col] < 0) | (cleaned[pf_col] > 100)
        )
        pf_invalid_counts[pf_col] = int(invalid_mask.sum())
        cleaned[f"{pf_col}_was_invalid"] = invalid_mask.astype(int)
        cleaned.loc[invalid_mask, pf_col] = float("nan")

    # Negative checks for non-negative physical quantities
    nonneg_cols = (
        "Usage_kWh",
        "Lagging_Current_Reactive.Power_kVarh",
        "Leading_Current_Reactive_Power_kVarh",
        "CO2(tCO2)",
    )
    negative_counts: dict[str, int] = {}
    for col in nonneg_cols:
        neg_mask = cleaned[col].notna() & (cleaned[col] < 0)
        negative_counts[col] = int(neg_mask.sum())
        if neg_mask.any():
            cleaned.loc[neg_mask, col] = float("nan")

    # NSM comparison against timestamp-derived NSM
    ts_nsm = compute_timestamp_nsm(cleaned["Date_Time"])
    nsm_present = cleaned["NSM"].notna()
    nsm_out_of_range = nsm_present & ((cleaned["NSM"] < 0) | (cleaned["NSM"] > 86400))
    nsm_mismatch = nsm_present & ((cleaned["NSM"] - ts_nsm).abs() > 1e-6)

    cleaned["NSM_was_inconsistent"] = nsm_mismatch.astype(int)
    cleaned.loc[nsm_mismatch, "NSM"] = float("nan")

    report: dict[str, Any] = {
        "input_rows": int(len(df)),
        "output_rows": int(len(cleaned)),
        "sorted_chronologically": bool(cleaned["Date_Time"].is_monotonic_increasing),
        "duplicate_timestamps_flagged": duplicate_ts_count,
        "power_factor_invalid_replaced_with_nan": pf_invalid_counts,
        "negative_measurements_replaced_with_nan": negative_counts,
        "nsm_audit": {
            "non_null_before_cleaning": int(nsm_present.sum()),
            "out_of_range_0_86400": int(nsm_out_of_range.sum()),
            "disagreed_with_timestamp": int(nsm_mismatch.sum()),
        },
        "missing_counts_after_cleaning": {
            col: int(cleaned[col].isna().sum()) for col in EXPECTED_COLUMNS
        },
    }
    return cleaned, report


def chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split cleaned data chronologically into train, validation, and held-out test sets.

    - Derives the final calendar month from Date_Time and reserves it as test (December).
    - Reserves the preceding calendar month as validation (November).
    - Uses all earlier records as training (January-October).
    - Guarantees equal timestamps stay in the same partition and no future-to-past overlap.
    """
    if not df["Date_Time"].is_monotonic_increasing:
        df = df.sort_values("Date_Time", kind="mergesort").reset_index(drop=True)

    periods = df["Date_Time"].dt.to_period("M")
    unique_periods = sorted(periods.unique())
    if len(unique_periods) < 3:
        raise ValueError(
            f"Need at least 3 distinct months for train/val/test split; found {len(unique_periods)}"
        )

    test_period = unique_periods[-1]
    val_period = unique_periods[-2]

    train_df = df.loc[periods < val_period].copy().reset_index(drop=True)
    val_df = df.loc[periods == val_period].copy().reset_index(drop=True)
    test_df = df.loc[periods == test_period].copy().reset_index(drop=True)

    # Verify strict chronological separation and zero timestamp overlap
    if not (train_df["Date_Time"].max() < val_df["Date_Time"].min()):
        raise AssertionError("Chronological boundary violation between train and val.")
    if not (val_df["Date_Time"].max() < test_df["Date_Time"].min()):
        raise AssertionError("Chronological boundary violation between val and test.")

    def _summarize(part: pd.DataFrame) -> dict[str, Any]:
        return {
            "row_count": int(len(part)),
            "start": part["Date_Time"].min().isoformat(),
            "end": part["Date_Time"].max().isoformat(),
            "class_counts": {
                cls: int((part["Load_Type"] == cls).sum()) for cls in EXPECTED_CLASSES
            },
        }

    summary: dict[str, Any] = {
        "train": _summarize(train_df),
        "val": _summarize(val_df),
        "test": _summarize(test_df),
        "dev_total_rows": int(len(train_df) + len(val_df)),
        "val_period": str(val_period),
        "test_period": str(test_period),
    }
    return train_df, val_df, test_df, summary


def save_audit_report(audit: dict[str, Any], output_path: str | Path) -> Path:
    """Write the data quality audit dictionary to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return out


def main() -> None:
    """Run Phase 1-2 data loading, audit, and deterministic cleaning report."""
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "data" / "load_data.csv"
    audit_path = project_root / "outputs" / "data_quality.json"

    df = load_raw_data(csv_path)
    audit = audit_dataset(df)
    _, cleaning_report = clean_dataset(df)
    audit["cleaning_report"] = cleaning_report
    save_audit_report(audit, audit_path)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
