"""Phase 10: Model loading, input validation, and single-row / batch CSV inference.

Security warning:
    Never load untrusted `.joblib` files. `joblib.load` uses Python pickle
    deserialization and can execute arbitrary code from untrusted sources.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.load_prediction.data import (
    EXPECTED_CLASSES,
    NUMERIC_COLUMNS,
    compute_timestamp_nsm,
)

INPUT_FEATURE_COLUMNS: tuple[str, ...] = (
    "Date_Time",
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
)


def load_trained_pipeline(model_path: str | Path) -> Pipeline:
    """Load a trusted fitted Pipeline from disk via joblib."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)


def prepare_inference_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate incoming inference rows and apply deterministic Phase 2 checks."""
    missing = [col for col in INPUT_FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    out = df[list(INPUT_FEATURE_COLUMNS)].copy()
    if out.empty:
        raise ValueError("Inference DataFrame is empty.")

    if out["Date_Time"].isna().any():
        raise ValueError("Date_Time contains missing values.")
    out["Date_Time"] = pd.to_datetime(out["Date_Time"], dayfirst=True, errors="raise")

    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="raise")

    for pf_col in ("Lagging_Current_Power_Factor", "Leading_Current_Power_Factor"):
        invalid_mask = out[pf_col].notna() & ((out[pf_col] < 0) | (out[pf_col] > 100))
        out[f"{pf_col}_was_invalid"] = invalid_mask.astype(int)
        out.loc[invalid_mask, pf_col] = float("nan")

    for col in (
        "Usage_kWh",
        "Lagging_Current_Reactive.Power_kVarh",
        "Leading_Current_Reactive_Power_kVarh",
        "CO2(tCO2)",
    ):
        neg_mask = out[col].notna() & (out[col] < 0)
        if neg_mask.any():
            out.loc[neg_mask, col] = float("nan")

    ts_nsm = compute_timestamp_nsm(out["Date_Time"])
    nsm_mismatch = out["NSM"].notna() & ((out["NSM"] - ts_nsm).abs() > 1e-6)
    out["NSM_was_inconsistent"] = nsm_mismatch.astype(int)
    out.loc[nsm_mismatch, "NSM"] = float("nan")

    return out


def predict_batch(
    data: pd.DataFrame | str | Path,
    model: Pipeline,
    class_names: tuple[str, ...] = EXPECTED_CLASSES,
) -> pd.DataFrame:
    """Run validated batch inference on a DataFrame or CSV path."""
    raw_df = pd.read_csv(data) if isinstance(data, (str, Path)) else data
    prepared = prepare_inference_dataframe(raw_df)

    sorted_classes = sorted(class_names)
    pred_ids = model.predict(prepared)
    pred_labels = [sorted_classes[int(idx)] for idx in pred_ids]

    result = pd.DataFrame(
        {
            "Date_Time": prepared["Date_Time"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_load_type": pred_labels,
        }
    )
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(prepared)
        for idx, cls_name in enumerate(sorted_classes):
            result[f"prob_{cls_name}"] = np.round(probs[:, idx], 6)

    return result


def predict_single(
    record: dict[str, Any],
    model: Pipeline,
    class_names: tuple[str, ...] = EXPECTED_CLASSES,
) -> dict[str, Any]:
    """Run validated single-record inference and return prediction dictionary."""
    df = pd.DataFrame([record])
    res_df = predict_batch(df, model=model, class_names=class_names)
    return res_df.iloc[0].to_dict()


def main() -> None:
    """CLI entry point for batch CSV prediction."""
    parser = argparse.ArgumentParser(description="Predict Power System Load Type from CSV.")
    parser.add_argument("--input", required=True, help="Path to input CSV.")
    parser.add_argument("--output", required=True, help="Path to write predictions CSV.")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parents[2] / "models" / "best_model.joblib"),
        help="Path to trusted best_model.joblib.",
    )
    args = parser.parse_args()

    model = load_trained_pipeline(args.model)
    preds = predict_batch(Path(args.input), model=model)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_path, index=False)
    print(f"Saved {len(preds)} predictions to {out_path}")


if __name__ == "__main__":
    main()
