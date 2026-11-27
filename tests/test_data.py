"""Phase 1 tests for data loading, schema validation, and audit reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.load_prediction.data import (
    EXPECTED_CLASSES,
    EXPECTED_COLUMNS,
    audit_dataset,
    chronological_split,
    clean_dataset,
    compute_timestamp_nsm,
    load_raw_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "load_data.csv"


def test_load_raw_data_contract() -> None:
    """Verify shape, columns, class counts, missingness, and day-first timestamps."""
    df = load_raw_data(DATA_PATH)
    audit = audit_dataset(df)

    assert audit["row_count"] == 35041
    assert audit["column_count"] == 9
    assert tuple(audit["columns"]) == EXPECTED_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(df["Date_Time"])

    assert audit["class_counts"] == {
        "Light_Load": 18073,
        "Medium_Load": 9696,
        "Maximum_Load": 7272,
    }
    assert set(df["Load_Type"].unique()) == set(EXPECTED_CLASSES)

    assert audit["missing_counts"] == {
        "Date_Time": 0,
        "Usage_kWh": 1559,
        "Lagging_Current_Reactive.Power_kVarh": 876,
        "Leading_Current_Reactive_Power_kVarh": 1156,
        "CO2(tCO2)": 455,
        "Lagging_Current_Power_Factor": 350,
        "Leading_Current_Power_Factor": 1471,
        "NSM": 455,
        "Load_Type": 0,
    }


def test_load_raw_data_missing_file(tmp_path: Path) -> None:
    """Loader raises FileNotFoundError for missing CSV."""
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "missing.csv")


def test_load_raw_data_schema_mismatch(tmp_path: Path) -> None:
    """Loader raises ValueError when columns do not match expected schema."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("Date_Time,Usage_kWh\n01-01-2018 00:15,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Schema mismatch"):
        load_raw_data(bad_csv)


def test_load_raw_data_malformed_timestamp(tmp_path: Path) -> None:
    """Loader raises ValueError on unparseable timestamps."""
    bad_csv = tmp_path / "bad_ts.csv"
    bad_csv.write_text(
        ",".join(EXPECTED_COLUMNS)
        + "\nnot-a-date,1.0,1.0,0.0,0.0,80.0,100.0,900.0,Light_Load\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_raw_data(bad_csv)


def test_clean_dataset_handles_anomalies_and_preserves_raw() -> None:
    """Cleaning sorts timestamps, flags invalid PF/NSM, and preserves extreme plausible Usage_kWh."""
    raw = pd.DataFrame(
        {
            "Date_Time": pd.to_datetime(
                ["2018-01-01 01:15:00", "2018-01-01 00:00:00", "2018-01-01 00:45:00"]
            ),
            "Usage_kWh": [435.0, 3.5, float("nan")],
            "Lagging_Current_Reactive.Power_kVarh": [10.0, 2.0, 3.0],
            "Leading_Current_Reactive_Power_kVarh": [0.0, 0.0, 0.0],
            "CO2(tCO2)": [0.18, 0.0, 0.0],
            "Lagging_Current_Power_Factor": [133.65, 75.0, -5.0],
            "Leading_Current_Power_Factor": [100.0, 210.0, 95.0],
            "NSM": [4500.0, 0.0, 8070.88],
            "Load_Type": ["Maximum_Load", "Light_Load", "Light_Load"],
        }
    )
    raw_snapshot = raw.copy(deep=True)
    cleaned, report = clean_dataset(raw)

    # Original DataFrame untouched
    pd.testing.assert_frame_equal(raw, raw_snapshot)

    # Chronologically sorted and no rows dropped
    assert report["input_rows"] == 3
    assert report["output_rows"] == 3
    assert cleaned["Date_Time"].is_monotonic_increasing

    # Extreme plausible Usage_kWh preserved
    assert cleaned.loc[cleaned["Date_Time"] == pd.Timestamp("2018-01-01 01:15:00"), "Usage_kWh"].iloc[0] == 435.0

    # Invalid power factors replaced with NaN and flagged
    assert cleaned["Lagging_Current_Power_Factor_was_invalid"].sum() == 2
    assert cleaned["Leading_Current_Power_Factor_was_invalid"].sum() == 1
    assert cleaned["Lagging_Current_Power_Factor"].isna().sum() == 2
    assert cleaned["Leading_Current_Power_Factor"].isna().sum() == 1

    # Inconsistent NSM flagged and replaced with NaN; consistent NSM preserved
    assert cleaned["NSM_was_inconsistent"].sum() == 1
    assert cleaned["NSM"].isna().sum() == 1
    assert compute_timestamp_nsm(cleaned["Date_Time"]).tolist() == [0.0, 2700.0, 4500.0]


def test_clean_dataset_on_real_data() -> None:
    """Verify deterministic cleaning on the full 35,041-row dataset."""
    df = load_raw_data(DATA_PATH)
    cleaned, report = clean_dataset(df)

    assert report["input_rows"] == 35041
    assert report["output_rows"] == 35041
    assert report["sorted_chronologically"] is True
    assert report["duplicate_timestamps_flagged"] == 1
    assert report["power_factor_invalid_replaced_with_nan"] == {
        "Lagging_Current_Power_Factor": 3226,
        "Leading_Current_Power_Factor": 2685,
    }
    assert report["nsm_audit"]["disagreed_with_timestamp"] == 3414
    assert cleaned["Date_Time"].is_monotonic_increasing


def test_chronological_split_boundaries_and_no_leakage() -> None:
    """Verify Jan-Oct train, Nov val, Dec test split with zero timestamp overlap."""
    df = load_raw_data(DATA_PATH)
    cleaned, _ = clean_dataset(df)
    train_df, val_df, test_df, summary = chronological_split(cleaned)

    assert len(train_df) + len(val_df) + len(test_df) == len(cleaned)
    assert summary["val_period"] == "2018-11"
    assert summary["test_period"] == "2018-12"

    # Strict chronological ordering across splits
    assert train_df["Date_Time"].max() < val_df["Date_Time"].min()
    assert val_df["Date_Time"].max() < test_df["Date_Time"].min()

    # Equal timestamps must stay in the same partition (e.g., 2018-12-31 20:00:00)
    train_ts = set(train_df["Date_Time"])
    val_ts = set(val_df["Date_Time"])
    test_ts = set(test_df["Date_Time"])
    assert train_ts.isdisjoint(val_ts)
    assert val_ts.isdisjoint(test_ts)
    assert train_ts.isdisjoint(test_ts)

    # All 3 classes must be present in each split
    for part_name in ("train", "val", "test"):
        counts = summary[part_name]["class_counts"]
        assert all(counts[cls] > 0 for cls in EXPECTED_CLASSES)
