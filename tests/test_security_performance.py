"""Phase 21 tests: security (no joblib/pickle uploads, CORS, limits) and performance (latency, memory)."""

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
DATA_PATH = PROJECT_ROOT / "data" / "load_data.csv"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Trained model not present.")
def test_security_rejects_serialized_models_and_enforces_cors_and_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://powerload-ai.vercel.app")
    svc = ModelInferenceService()
    svc.load()
    app = create_app(service=svc)

    async def _run() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            for bad_name in ("evil.joblib", "evil.pkl", "evil.pickle"):
                resp = await client.post(
                    "/api/v1/predict/batch",
                    files={"file": (bad_name, b"pickle-payload", "application/octet-stream")},
                )
                assert resp.status_code == 415

            ok_cors = await client.options(
                "/api/v1/health",
                headers={
                    "Origin": "https://powerload-ai.vercel.app",
                    "Access-Control-Request-Method": "GET",
                },
            )
            assert ok_cors.headers.get("access-control-allow-origin") == "https://powerload-ai.vercel.app"

            bad_cors = await client.options(
                "/api/v1/health",
                headers={
                    "Origin": "https://untrusted.example.com",
                    "Access-Control-Request-Method": "GET",
                },
            )
            assert "access-control-allow-origin" not in bad_cors.headers

            extra_resp = await client.post(
                "/api/v1/predict",
                json={
                    "Date_Time": "15-12-2018 10:30",
                    "Usage_kWh": 10.0,
                    "unexpected_field": 123,
                },
            )
            assert extra_resp.status_code == 422

    asyncio.run(_run())


@pytest.mark.skipif(
    not (MODEL_PATH.exists() and DATA_PATH.exists()),
    reason="Model or dataset missing.",
)
def test_performance_latency_memory_and_batch_bounds() -> None:
    svc = ModelInferenceService()
    svc.load()
    health = svc.get_health()
    assert health["memory_rss_mb"] < 1024.0

    raw = pd.read_csv(DATA_PATH)
    single_row = raw.iloc[-1].drop("Load_Type").to_dict()
    single_res = svc.predict_one(single_row)
    assert single_res["inference_time_ms"] < 1000.0

    batch_df = raw.iloc[-250:].drop(columns=["Load_Type"])
    batch_res = svc.predict_batch_df(batch_df)
    assert batch_res["total_rows"] == 250
    assert batch_res["inference_time_ms"] < 2000.0
