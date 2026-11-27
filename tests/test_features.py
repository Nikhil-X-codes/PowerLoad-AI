"""Phase 5 tests for feature derivation, preprocessing pipelines, and leakage prevention."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.load_prediction.data import chronological_split, clean_dataset, load_raw_data
from src.load_prediction.features import (
    build_preprocessor,
    derive_features,
    get_feature_columns,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "load_data.csv"


def test_derive_features_drops_target_and_computes_expected_columns() -> None:
    """Derived features exclude Load_Type and compute cyclic hour and reactive magnitude."""
    df = pd.DataFrame(
        {
            "Date_Time": pd.to_datetime(["2018-01-06 06:00:00", "2018-01-08 18:00:00"]),
            "Usage_kWh": [10.0, 20.0],
            "Lagging_Current_Reactive.Power_kVarh": [3.0, 6.0],
            "Leading_Current_Reactive_Power_kVarh": [4.0, 8.0],
            "CO2(tCO2)": [0.01, 0.02],
            "Lagging_Current_Power_Factor": [80.0, 90.0],
            "Leading_Current_Power_Factor": [100.0, 100.0],
            "NSM": [21600.0, 64800.0],
            "Load_Type": ["Light_Load", "Maximum_Load"],
        }
    )
    feat = derive_features(df)
    assert "Load_Type" not in feat.columns
    assert feat["is_weekend"].tolist() == [1.0, 0.0]  # 2018-01-06 is Saturday, 01-08 is Monday
    assert np.allclose(feat["total_reactive_power_magnitude"].to_numpy(), [5.0, 10.0])
    assert np.allclose(feat["hour_sin"].to_numpy(), [1.0, -1.0], atol=1e-6)
    assert np.allclose(feat["hour_cos"].to_numpy(), [0.0, 0.0], atol=1e-6)


@pytest.mark.parametrize("feature_set", ["timestamp_only", "measurement_only", "combined"])
def test_preprocessor_train_and_inference_match(feature_set: str) -> None:
    """Fitted preprocessor produces identical shapes and finite values on train and val."""
    raw = load_raw_data(DATA_PATH)
    cleaned, _ = clean_dataset(raw)
    train_df, val_df, _, _ = chronological_split(cleaned)

    prep = build_preprocessor(feature_set=feature_set, scale_numeric=True)  # type: ignore[arg-type]
    X_train = prep.fit_transform(train_df)
    X_val = prep.transform(val_df)

    assert X_train.shape[0] == len(train_df)
    assert X_val.shape[0] == len(val_df)
    assert X_train.shape[1] == X_val.shape[1]
    assert np.isfinite(X_train).all()
    assert np.isfinite(X_val).all()


def test_get_feature_columns_invalid_raises() -> None:
    """Unknown feature_set raises ValueError."""
    with pytest.raises(ValueError, match="Unknown feature_set"):
        get_feature_columns("invalid_set")  # type: ignore[arg-type]
