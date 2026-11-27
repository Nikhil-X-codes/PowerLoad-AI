"""Phase 13 tests: centralized ModelInferenceService, parity with notebook outputs, latency & memory."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.load_prediction.service import ModelInferenceService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "load_data.csv"
PREDS_PATH = PROJECT_ROOT / "outputs" / "predictions.csv"


@pytest.mark.skipif(
    not (MODEL_PATH.exists() and DATA_PATH.exists() and PREDS_PATH.exists()),
    reason="Production model or dataset artifacts missing.",
)
def test_service_loads_once_and_matches_saved_notebook_predictions() -> None:
    svc = ModelInferenceService()
    svc.load()
    first_load_ms = svc.load_time_ms
    assert svc.is_loaded is True
    assert first_load_ms is not None and first_load_ms >= 0.0

    # Calling load() again is a no-op
    svc.load()
    assert svc.load_time_ms == first_load_ms

    health = svc.get_health()
    assert health["status"] == "ok"
    assert health["model_version"] == "1.0.0"
    assert health["memory_rss_mb"] > 0

    info = svc.get_model_info()
    assert info["model_name"] == "RandomForest"
    assert info["classes"] == ["Light_Load", "Maximum_Load", "Medium_Load"]
    assert len(info["input_features"]) == 8

    # Verify batch and single-record parity against outputs/predictions.csv
    raw = pd.read_csv(DATA_PATH)
    dec_slice = raw.iloc[-50:-25].drop(columns=["Load_Type"])
    batch_res = svc.predict_batch_df(dec_slice)
    assert batch_res["total_rows"] == 25
    assert batch_res["inference_time_ms"] >= 0.0

    saved_df = pd.read_csv(PREDS_PATH)
    pred_df = pd.DataFrame(batch_res["predictions"]).merge(
        saved_df, on="Date_Time", how="inner"
    )
    assert len(pred_df) == 25
    assert (pred_df["prediction"] == pred_df["predicted_load_type"]).all()

    # Single-record parity check including probability match
    sample_row = dec_slice.iloc[0].to_dict()
    single_res = svc.predict_one(sample_row)
    assert single_res["prediction"] == pred_df.iloc[0]["predicted_load_type"]
    assert single_res["inference_time_ms"] >= 0.0
    assert abs(sum(single_res["probabilities"].values()) - 1.0) < 1e-3
    for cls_name in info["classes"]:
        assert abs(single_res["probabilities"][cls_name] - pred_df.iloc[0][f"prob_{cls_name}"]) < 1e-3

