"""Phases 8-9 tests for final evaluation metrics recomputability and explainability."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_final_metrics_recomputable_from_predictions() -> None:
    """Verify outputs/metrics.json matches recomputed metrics from outputs/predictions.csv."""
    pred_path = PROJECT_ROOT / "outputs" / "predictions.csv"
    metrics_path = PROJECT_ROOT / "outputs" / "metrics.json"
    if not (pred_path.exists() and metrics_path.exists()):
        return

    pred_df = pd.read_csv(pred_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert len(pred_df) == metrics["test_sample_count"] == 2977
    recomputed_acc = round(
        float(accuracy_score(pred_df["actual_load_type"], pred_df["predicted_load_type"])), 4
    )
    recomputed_f1 = round(
        float(
            f1_score(
                pred_df["actual_load_type"],
                pred_df["predicted_load_type"],
                average="macro",
                zero_division=0,
            )
        ),
        4,
    )
    assert recomputed_acc == metrics["summary_metrics"]["accuracy"]
    assert recomputed_f1 == metrics["summary_metrics"]["macro_f1"]

    cm_total = sum(sum(row) for row in metrics["confusion_matrix"]["matrix"])
    assert cm_total == metrics["test_sample_count"]
