"""Phases 6-7 tests for candidate model evaluation, metrics, and chronological CV."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.load_prediction.data import EXPECTED_CLASSES
from src.load_prediction.train import (
    compute_classification_metrics,
    evaluate_candidates,
    get_label_encoder,
    tune_top_models,
)


def _make_synthetic_df(n_rows: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-01", periods=n_rows, freq="15min")
    usage = rng.uniform(1.0, 120.0, size=n_rows)
    labels = np.where(
        usage < 25.0,
        "Light_Load",
        np.where(usage < 70.0, "Medium_Load", "Maximum_Load"),
    )
    return pd.DataFrame(
        {
            "Date_Time": dates,
            "Usage_kWh": usage,
            "Lagging_Current_Reactive.Power_kVarh": rng.uniform(0.0, 40.0, size=n_rows),
            "Leading_Current_Reactive_Power_kVarh": rng.uniform(0.0, 20.0, size=n_rows),
            "CO2(tCO2)": usage * 0.0005,
            "Lagging_Current_Power_Factor": rng.uniform(50.0, 100.0, size=n_rows),
            "Leading_Current_Power_Factor": rng.uniform(50.0, 100.0, size=n_rows),
            "NSM": (dates.hour * 3600 + dates.minute * 60).astype(float),
            "Load_Type": labels,
        }
    )


def test_label_encoder_canonical_order() -> None:
    """LabelEncoder uses the canonical EXPECTED_CLASSES order."""
    le = get_label_encoder()
    assert set(le.classes_) == set(EXPECTED_CLASSES)
    assert len(le.classes_) == 3


def test_compute_classification_metrics_zero_division() -> None:
    """Metrics handle missing predicted classes without warnings or NaNs."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 0, 0, 0, 0, 0])
    m = compute_classification_metrics(
        y_true, y_pred, labels=[0, 1, 2], label_names=list(EXPECTED_CLASSES)
    )
    assert np.isfinite(m["macro_f1"])
    assert np.isfinite(m["accuracy"])
    assert m["accuracy"] == 2 / 6


def test_evaluate_candidates_and_tune_on_synthetic() -> None:
    """Candidate comparison and TimeSeriesSplit tuning run end-to-end."""
    df = _make_synthetic_df(180)
    train_df = df.iloc[:120].copy()
    val_df = df.iloc[120:].copy()

    comp = evaluate_candidates(train_df, val_df, feature_set="combined")
    assert "DummyClassifier" in set(comp["model"])
    assert "RandomForest" in set(comp["model"])
    assert "LightGBM" in set(comp["model"])
    assert comp.loc[comp["model"] == "DummyClassifier", "status"].iloc[0] == "ok"

    tuning = tune_top_models(df, n_splits=3, n_iter=2)
    assert tuning["primary_metric"] == "f1_macro"
    assert tuning["winner"]["best_cv_macro_f1"] > 0.0
