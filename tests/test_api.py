"""Phase 14 integration tests for FastAPI v1 endpoints, validation, CORS, and Swagger docs."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pandas as pd
import pytest

from src.load_prediction.api import create_app
from src.load_prediction.service import ModelInferenceService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Trained model not present.")
def test_api_v1_endpoints_end_to_end() -> None:
    svc = ModelInferenceService()
    svc.load()
    app = create_app(service=svc)

    async def _run() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            h = await client.get("/api/v1/health")
            assert h.status_code == 200
            h_body = h.json()
            assert h_body["status"] == "ok"
            assert h_body["model_loaded"] is True
            assert h_body["model_version"] == "1.0.0"

            info = await client.get("/api/v1/model/info")
            assert info.status_code == 200
            info_body = info.json()
            assert info_body["model_name"] == "RandomForest"
            assert "input_schema" in info_body

            payload = {
                "Date_Time": "15-12-2018 10:30",
                "Usage_kWh": 38.5,
                "Lagging_Current_Reactive.Power_kVarh": 12.5,
                "Leading_Current_Reactive_Power_kVarh": 0.0,
                "CO2(tCO2)": 0.02,
                "Lagging_Current_Power_Factor": 85.0,
                "Leading_Current_Power_Factor": 100.0,
                "NSM": 37800.0,
            }
            pred = await client.post("/api/v1/predict", json=payload)
            assert pred.status_code == 200
            pred_body = pred.json()
            assert pred_body["prediction"] in {"Light_Load", "Medium_Load", "Maximum_Load"}
            assert abs(sum(pred_body["probabilities"].values()) - 1.0) < 1e-3
            assert pred_body["model_version"] == "1.0.0"
            assert pred_body["inference_time_ms"] >= 0.0

            bad_pred = await client.post(
                "/api/v1/predict",
                json={**payload, "Date_Time": "not-a-date"},
            )
            assert bad_pred.status_code == 422

            csv_df = pd.DataFrame([payload, {**payload, "Date_Time": "15-12-2018 10:45"}])
            csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
            batch = await client.post(
                "/api/v1/predict/batch",
                files={"file": ("sample.csv", csv_bytes, "text/csv")},
            )
            assert batch.status_code == 200
            batch_body = batch.json()
            assert batch_body["total_rows"] == 2
            assert len(batch_body["predictions"]) == 2
            assert sum(batch_body["class_distribution"].values()) == 2

            bad_ext = await client.post(
                "/api/v1/predict/batch",
                files={"file": ("sample.txt", b"hello", "text/plain")},
            )
            assert bad_ext.status_code == 415

            empty_csv = await client.post(
                "/api/v1/predict/batch",
                files={"file": ("empty.csv", b"", "text/csv")},
            )
            assert empty_csv.status_code == 400

            missing_cols = await client.post(
                "/api/v1/predict/batch",
                files={"file": ("bad.csv", b"Date_Time,Usage_kWh\n15-12-2018 10:30,10.0\n", "text/csv")},
            )
            assert missing_cols.status_code == 422

            metrics = await client.get("/api/v1/metrics")
            assert metrics.status_code == 200
            assert metrics.json()["summary_metrics"]["accuracy"] == 0.9332

            analytics = await client.get("/api/v1/analytics")
            assert analytics.status_code == 200
            a_body = analytics.json()
            assert "data_quality" in a_body
            assert "model_comparison" in a_body
            assert "historical_summary" in a_body

            docs = await client.get("/docs")
            assert docs.status_code == 200
            openapi = await client.get("/openapi.json")
            assert openapi.status_code == 200

    asyncio.run(_run())
