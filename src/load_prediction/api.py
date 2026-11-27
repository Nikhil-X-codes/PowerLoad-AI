"""Phase 14: FastAPI REST backend with Swagger docs, validation, CORS, and structured logging."""

from __future__ import annotations

import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.load_prediction.service import ModelInferenceService

MAX_CSV_UPLOAD_BYTES = int(os.environ.get("MAX_CSV_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_BATCH_ROWS = int(os.environ.get("MAX_BATCH_ROWS", "50000"))


class _JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            log_record.update(extra)
        return json.dumps(log_record)


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger("load_prediction.api")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonLogFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


logger = _configure_logger()


class LoadRecordRequest(BaseModel):
    """Validated single-record input matching the raw power system measurement schema."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    Date_Time: str = Field(
        ...,
        description="Timestamp in DD-MM-YYYY HH:MM or ISO format.",
        examples=["15-12-2018 10:30"],
    )
    Usage_kWh: float | None = Field(default=None, examples=[35.5])
    Lagging_Current_Reactive_Power_kVarh: float | None = Field(
        default=None,
        alias="Lagging_Current_Reactive.Power_kVarh",
        examples=[12.5],
    )
    Leading_Current_Reactive_Power_kVarh: float | None = Field(default=None, examples=[0.0])
    CO2_tCO2: float | None = Field(default=None, alias="CO2(tCO2)", examples=[0.02])
    Lagging_Current_Power_Factor: float | None = Field(default=None, examples=[85.0])
    Leading_Current_Power_Factor: float | None = Field(default=None, examples=[100.0])
    NSM: float | None = Field(default=None, examples=[37800.0])

    def to_raw_record(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)


class PredictionResponse(BaseModel):
    Date_Time: str
    prediction: str
    probabilities: dict[str, float]
    model_version: str
    inference_time_ms: float


class BatchPredictionRow(BaseModel):
    Date_Time: str
    prediction: str
    probabilities: dict[str, float]


class BatchPredictionResponse(BaseModel):
    total_rows: int
    model_version: str
    inference_time_ms: float
    class_distribution: dict[str, int]
    predictions: list[BatchPredictionRow]


def create_app(service: ModelInferenceService | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    inference_service = service or ModelInferenceService()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = inference_service
        try:
            inference_service.load()
            logger.info(
                "Model loaded on startup",
                extra={"extra_fields": inference_service.get_health()},
            )
        except Exception as exc:
            logger.error(
                "Startup model load failed",
                extra={"extra_fields": {"error": str(exc)}},
            )
        yield

    app = FastAPI(
        title="Power System Load Type Prediction API",
        version="1.0.0",
        description=(
            "Production REST API for predicting power system load types "
            "(Light_Load, Medium_Load, Maximum_Load) with calibrated probabilities and analytics."
        ),
        lifespan=lifespan,
    )
    app.state.service = inference_service

    allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000.0, 2)
        logger.info(
            "HTTP request",
            extra={
                "extra_fields": {
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                }
            },
        )
        return response

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": str(exc)},
        )

    @app.get("/api/v1/health")
    def health(request: Request) -> dict[str, Any]:
        svc: ModelInferenceService = request.app.state.service
        if not svc.is_loaded:
            try:
                svc.load()
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Model unavailable: {exc}",
                ) from exc
        return svc.get_health()

    @app.get("/api/v1/model/info")
    def model_info(request: Request) -> dict[str, Any]:
        svc: ModelInferenceService = request.app.state.service
        try:
            info = svc.get_model_info()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        info["input_schema"] = LoadRecordRequest.model_json_schema()
        return info

    @app.post("/api/v1/predict", response_model=PredictionResponse)
    def predict_endpoint(
        request: Request, payload: LoadRecordRequest
    ) -> dict[str, Any]:
        svc: ModelInferenceService = request.app.state.service
        try:
            return svc.predict_one(payload.to_raw_record())
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

    @app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse)
    async def predict_batch_endpoint(
        request: Request,
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        svc: ModelInferenceService = request.app.state.service
        filename = (file.filename or "").strip()
        if not filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only .csv files are supported for batch prediction.",
            )

        try:
            raw_bytes = await file.read()
        finally:
            await file.close()

        if not raw_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded CSV file is empty (0 bytes).",
            )
        if len(raw_bytes) > MAX_CSV_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"CSV exceeds maximum size ({len(raw_bytes)} > {MAX_CSV_UPLOAD_BYTES} bytes).",
            )

        try:
            df = pd.read_csv(io.BytesIO(raw_bytes))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Failed to parse CSV file: {exc}",
            ) from exc

        if len(df) > MAX_BATCH_ROWS:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"CSV row count ({len(df)}) exceeds limit ({MAX_BATCH_ROWS}).",
            )

        try:
            return svc.predict_batch_df(df)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

    @app.get("/api/v1/metrics")
    def metrics_endpoint(request: Request) -> dict[str, Any]:
        svc: ModelInferenceService = request.app.state.service
        try:
            return svc.get_metrics()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc

    @app.get("/api/v1/analytics")
    def analytics_endpoint(request: Request) -> dict[str, Any]:
        svc: ModelInferenceService = request.app.state.service
        return svc.get_analytics()

    return app


app = create_app()

