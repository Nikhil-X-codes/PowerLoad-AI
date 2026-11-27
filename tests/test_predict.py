"""Phases 10-11 tests for model persistence, inference validation, and consistency."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.load_prediction.data import EXPECTED_CLASSES
from src.load_prediction.predict import (
    load_trained_pipeline,
    predict_batch,
    predict_single,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "load_data.csv"
PREDS_PATH = PROJECT_ROOT / "outputs" / "predictions.csv"


def test_model_reload_matches_saved_predictions() -> None:
    """Reloaded best_model.joblib produces identical predictions on a December fixture."""
    if not (MODEL_PATH.exists() and PREDS_PATH.exists()):
        pytest.skip("Trained model or saved predictions not present.")

    model = load_trained_pipeline(MODEL_PATH)
    raw = pd.read_csv(DATA_PATH)
    dec_raw = raw.iloc[-100:-75].drop(columns=["Load_Type"])

    batch_preds = predict_batch(dec_raw, model=model)
    saved_preds = pd.read_csv(PREDS_PATH)

    merged = batch_preds.merge(saved_preds, on="Date_Time", suffixes=("_reloaded", "_saved"))
    assert len(merged) == 25
    assert (merged["predicted_load_type_reloaded"] == merged["predicted_load_type_saved"]).all()


def test_predict_single_with_missing_and_invalid_values() -> None:
    """Single-record inference handles missing Usage_kWh and invalid power factors gracefully."""
    if not MODEL_PATH.exists():
        pytest.skip("Trained model not present.")

    model = load_trained_pipeline(MODEL_PATH)
    record = {
        "Date_Time": "15-12-2018 10:30",
        "Usage_kWh": None,
        "Lagging_Current_Reactive.Power_kVarh": 12.5,
        "Leading_Current_Reactive_Power_kVarh": 0.0,
        "CO2(tCO2)": 0.01,
        "Lagging_Current_Power_Factor": 145.0,
        "Leading_Current_Power_Factor": 100.0,
        "NSM": 99999.0,
    }
    result = predict_single(record, model=model)
    assert result["predicted_load_type"] in EXPECTED_CLASSES
    assert result["Date_Time"] == "2018-12-15 10:30:00"


def test_predict_rejects_missing_columns_and_bad_timestamp(tmp_path: Path) -> None:
    """Inference raises ValueError on missing schema columns or invalid timestamps."""
    if not MODEL_PATH.exists():
        pytest.skip("Trained model not present.")

    model = load_trained_pipeline(MODEL_PATH)

    bad_cols = pd.DataFrame([{"Date_Time": "01-12-2018 00:15", "Usage_kWh": 5.0}])
    with pytest.raises(ValueError, match="Missing required input columns"):
        predict_batch(bad_cols, model=model)

    bad_ts_csv = tmp_path / "bad_ts.csv"
    bad_ts_csv.write_text(
        "Date_Time,Usage_kWh,Lagging_Current_Reactive.Power_kVarh,"
        "Leading_Current_Reactive_Power_kVarh,CO2(tCO2),"
        "Lagging_Current_Power_Factor,Leading_Current_Power_Factor,NSM\n"
        "invalid-date,5.0,2.0,0.0,0.0,80.0,100.0,900.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        predict_batch(bad_ts_csv, model=model)
