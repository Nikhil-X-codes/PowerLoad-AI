"""Phase 13: Centralized production model inference and analytics service.

Security warning:
    Never load untrusted `.joblib` files. `joblib.load` uses Python pickle
    deserialization and can execute arbitrary code from untrusted sources.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
from sklearn.pipeline import Pipeline

from src.load_prediction.data import EXPECTED_CLASSES, load_raw_data
from src.load_prediction.features import (
    ANOMALY_INDICATOR_FEATURES,
    MEASUREMENT_NUMERIC_FEATURES,
    TIMESTAMP_FEATURES,
)
from src.load_prediction.predict import (
    INPUT_FEATURE_COLUMNS,
    load_trained_pipeline,
    predict_batch,
    predict_single,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics.json"
DEFAULT_DATA_QUALITY_PATH = PROJECT_ROOT / "outputs" / "data_quality.json"
DEFAULT_MODEL_COMPARISON_PATH = PROJECT_ROOT / "outputs" / "model_comparison.csv"
DEFAULT_EXPLAINABILITY_PATH = PROJECT_ROOT / "outputs" / "explainability.json"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "load_data.csv"

MODEL_VERSION = "1.0.0"


def get_process_memory_mb() -> float:
    """Return current process resident set size (RSS) in megabytes."""
    rss_bytes = psutil.Process(os.getpid()).memory_info().rss
    return round(rss_bytes / (1024.0 * 1024.0), 2)


@dataclass
class ModelInferenceService:
    """Centralized singleton-friendly service that loads the trained pipeline once."""

    model_path: Path = DEFAULT_MODEL_PATH
    metadata_path: Path = DEFAULT_METADATA_PATH
    metrics_path: Path = DEFAULT_METRICS_PATH
    data_quality_path: Path = DEFAULT_DATA_QUALITY_PATH
    model_comparison_path: Path = DEFAULT_MODEL_COMPARISON_PATH
    explainability_path: Path = DEFAULT_EXPLAINABILITY_PATH
    dataset_path: Path = DEFAULT_DATASET_PATH
    model_version: str = MODEL_VERSION

    model: Pipeline | None = field(default=None, init=False, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict, init=False)
    load_time_ms: float | None = field(default=None, init=False)
    loaded_at_monotonic: float | None = field(default=None, init=False)
    _analytics_cache: dict[str, Any] | None = field(default=None, init=False, repr=False)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        """Load the trained pipeline and metadata once during startup."""
        if self.model is not None:
            return
        start = time.perf_counter()
        self.model = load_trained_pipeline(self.model_path)
        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        else:
            self.metadata = {
                "model_name": type(self.model.named_steps.get("model", self.model)).__name__,
                "feature_set": "combined",
                "classes": sorted(EXPECTED_CLASSES),
            }
        self.load_time_ms = round((time.perf_counter() - start) * 1000.0, 2)
        self.loaded_at_monotonic = time.monotonic()

    def ensure_loaded(self) -> Pipeline:
        if self.model is None:
            self.load()
        assert self.model is not None
        return self.model

    def get_health(self) -> dict[str, Any]:
        uptime = (
            round(time.monotonic() - self.loaded_at_monotonic, 2)
            if self.loaded_at_monotonic is not None
            else 0.0
        )
        return {
            "status": "ok" if self.is_loaded else "not_loaded",
            "model_loaded": self.is_loaded,
            "model_version": self.model_version,
            "model_name": self.metadata.get("model_name", "RandomForest"),
            "load_time_ms": self.load_time_ms,
            "memory_rss_mb": get_process_memory_mb(),
            "uptime_seconds": uptime,
        }

    def get_model_info(self) -> dict[str, Any]:
        self.ensure_loaded()
        return {
            "model_version": self.model_version,
            "model_name": self.metadata.get("model_name", "RandomForest"),
            "feature_set": self.metadata.get("feature_set", "combined"),
            "best_params": self.metadata.get("best_params", {}),
            "classes": self.metadata.get("classes", sorted(EXPECTED_CLASSES)),
            "train_rows": self.metadata.get("train_rows"),
            "input_features": list(INPUT_FEATURE_COLUMNS),
            "engineered_features": {
                "timestamp_features": list(TIMESTAMP_FEATURES),
                "measurement_features": list(MEASUREMENT_NUMERIC_FEATURES),
                "anomaly_indicator_features": list(ANOMALY_INDICATOR_FEATURES),
            },
            "security_note": self.metadata.get(
                "security_note",
                "Never load untrusted .joblib files; joblib uses pickle deserialization.",
            ),
        }

    def predict_one(self, record: dict[str, Any]) -> dict[str, Any]:
        """Run single-record inference with latency measurement."""
        pipeline = self.ensure_loaded()
        classes = tuple(self.metadata.get("classes", sorted(EXPECTED_CLASSES)))

        start = time.perf_counter()
        raw_result = predict_single(record, model=pipeline, class_names=classes)
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)

        probabilities = {
            cls_name: round(float(raw_result[f"prob_{cls_name}"]), 4)
            for cls_name in sorted(classes)
            if f"prob_{cls_name}" in raw_result
        }
        return {
            "Date_Time": raw_result["Date_Time"],
            "prediction": raw_result["predicted_load_type"],
            "probabilities": probabilities,
            "model_version": self.model_version,
            "inference_time_ms": elapsed_ms,
        }

    def predict_batch_df(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run validated batch inference on a DataFrame and return summary + rows."""
        pipeline = self.ensure_loaded()
        classes = tuple(self.metadata.get("classes", sorted(EXPECTED_CLASSES)))

        start = time.perf_counter()
        res_df = predict_batch(df, model=pipeline, class_names=classes)
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)

        sorted_classes = sorted(classes)
        rows: list[dict[str, Any]] = []
        for rec in res_df.to_dict(orient="records"):
            probs = {
                cls_name: round(float(rec[f"prob_{cls_name}"]), 4)
                for cls_name in sorted_classes
                if f"prob_{cls_name}" in rec
            }
            rows.append(
                {
                    "Date_Time": rec["Date_Time"],
                    "prediction": rec["predicted_load_type"],
                    "probabilities": probs,
                }
            )

        counts = res_df["predicted_load_type"].value_counts().to_dict()
        class_distribution = {cls_name: int(counts.get(cls_name, 0)) for cls_name in sorted_classes}

        return {
            "total_rows": len(rows),
            "model_version": self.model_version,
            "inference_time_ms": elapsed_ms,
            "class_distribution": class_distribution,
            "predictions": rows,
        }

    def get_metrics(self) -> dict[str, Any]:
        """Return actual Phase 8 evaluation metrics from outputs/metrics.json."""
        if not self.metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {self.metrics_path}")
        return json.loads(self.metrics_path.read_text(encoding="utf-8"))

    def get_analytics(self) -> dict[str, Any]:
        """Return historical dataset summaries, model comparison, and feature explainability."""
        if self._analytics_cache is not None:
            return self._analytics_cache

        payload: dict[str, Any] = {"model_version": self.model_version}

        if self.data_quality_path.exists():
            payload["data_quality"] = json.loads(
                self.data_quality_path.read_text(encoding="utf-8")
            )
        if self.model_comparison_path.exists():
            comp_df = pd.read_csv(self.model_comparison_path)
            comp_df = comp_df.astype(object).where(comp_df.notna(), None)
            payload["model_comparison"] = comp_df.to_dict(orient="records")
        if self.explainability_path.exists():
            payload["explainability"] = json.loads(
                self.explainability_path.read_text(encoding="utf-8")
            )

        if self.dataset_path.exists():
            df = load_raw_data(self.dataset_path)
            dt = pd.to_datetime(df["Date_Time"], dayfirst=True)
            df = df.assign(hour=dt.dt.hour, month=dt.dt.month)

            hourly = (
                df.groupby(["hour", "Load_Type"])
                .size()
                .unstack(fill_value=0)
                .reset_index()
                .to_dict(orient="records")
            )
            monthly = (
                df.groupby(["month", "Load_Type"])
                .size()
                .unstack(fill_value=0)
                .reset_index()
                .to_dict(orient="records")
            )
            usage_summary = (
                df.groupby("Load_Type")["Usage_kWh"]
                .agg(["count", "mean", "median", "min", "max"])
                .round(3)
                .reset_index()
                .to_dict(orient="records")
            )
            payload["historical_summary"] = {
                "hourly_load_distribution": hourly,
                "monthly_load_distribution": monthly,
                "usage_kwh_by_load_type": usage_summary,
            }

        self._analytics_cache = payload
        return payload

