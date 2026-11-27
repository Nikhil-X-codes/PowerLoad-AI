"""Phase 5: Deterministic feature engineering and scikit-learn preprocessing pipelines."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.load_prediction.data import compute_timestamp_nsm

FeatureSetName = Literal["timestamp_only", "measurement_only", "combined"]

TIMESTAMP_FEATURES: tuple[str, ...] = (
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "hour_sin",
    "hour_cos",
    "ts_nsm",
)

MEASUREMENT_NUMERIC_FEATURES: tuple[str, ...] = (
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
    "total_reactive_power_magnitude",
)

ANOMALY_INDICATOR_FEATURES: tuple[str, ...] = (
    "Lagging_Current_Power_Factor_was_invalid",
    "Leading_Current_Power_Factor_was_invalid",
    "NSM_was_inconsistent",
)


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive deterministic time and reactive-power features without target access."""
    out = df.copy()
    if "Load_Type" in out.columns:
        out = out.drop(columns=["Load_Type"])

    dt = pd.to_datetime(out["Date_Time"], dayfirst=True, errors="raise")
    hour = dt.dt.hour.astype(float)
    day_of_week = dt.dt.dayofweek.astype(float)

    out["hour"] = hour
    out["day_of_week"] = day_of_week
    out["is_weekend"] = (day_of_week >= 5).astype(float)
    out["month"] = dt.dt.month.astype(float)

    angle = 2.0 * np.pi * hour / 24.0
    out["hour_sin"] = np.sin(angle)
    out["hour_cos"] = np.cos(angle)
    out["ts_nsm"] = compute_timestamp_nsm(dt)

    lag_q = pd.to_numeric(
        out["Lagging_Current_Reactive.Power_kVarh"], errors="coerce"
    )
    lead_q = pd.to_numeric(
        out["Leading_Current_Reactive_Power_kVarh"], errors="coerce"
    )
    out["total_reactive_power_magnitude"] = np.hypot(lag_q, lead_q)

    for flag_col in ANOMALY_INDICATOR_FEATURES:
        if flag_col not in out.columns:
            out[flag_col] = 0.0
        else:
            out[flag_col] = out[flag_col].astype(float)

    return out


def get_feature_columns(feature_set: FeatureSetName) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, indicator_cols) for the requested feature set."""
    if feature_set == "timestamp_only":
        return list(TIMESTAMP_FEATURES), []
    if feature_set == "measurement_only":
        return list(MEASUREMENT_NUMERIC_FEATURES), list(ANOMALY_INDICATOR_FEATURES)
    if feature_set == "combined":
        return (
            list(TIMESTAMP_FEATURES) + list(MEASUREMENT_NUMERIC_FEATURES),
            list(ANOMALY_INDICATOR_FEATURES),
        )
    raise ValueError(f"Unknown feature_set: {feature_set!r}")


def build_preprocessor(
    feature_set: FeatureSetName = "combined",
    scale_numeric: bool = True,
) -> Pipeline:
    """Build a reusable sklearn Pipeline for feature derivation, imputation, and scaling."""
    numeric_cols, indicator_cols = get_feature_columns(feature_set)

    num_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, object, list[str]]] = [
        ("num", Pipeline(num_steps), numeric_cols),
    ]
    if indicator_cols:
        transformers.append(
            (
                "flags",
                SimpleImputer(strategy="constant", fill_value=0.0),
                indicator_cols,
            )
        )

    col_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        [
            ("derive", FunctionTransformer(derive_features, validate=False)),
            ("preprocess", col_transformer),
        ]
    )
