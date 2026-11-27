"""Phases 6-7: Candidate model training, chronological validation, and hyperparameter tuning."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from src.load_prediction.data import (
    EXPECTED_CLASSES,
    chronological_split,
    clean_dataset,
    load_raw_data,
)
from src.load_prediction.features import FeatureSetName, build_preprocessor

RANDOM_SEED = 42


def get_label_encoder() -> LabelEncoder:
    """Return a LabelEncoder fitted on the canonical EXPECTED_CLASSES order."""
    le = LabelEncoder()
    le.fit(list(EXPECTED_CLASSES))
    return le


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    labels: list[int] | list[str] | None = None,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute accuracy, macro/weighted precision, recall, F1, and per-class F1."""
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if labels is not None and label_names is not None:
        per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        for name, score in zip(label_names, per_class_f1):
            metrics[f"f1_{name}"] = float(score)
    return metrics


def get_candidate_specs(random_state: int = RANDOM_SEED) -> list[dict[str, Any]]:
    """Return Phase 6 candidate model specifications."""
    specs: list[dict[str, Any]] = [
        {
            "name": "DummyClassifier",
            "estimator": DummyClassifier(strategy="most_frequent"),
            "scale_numeric": False,
            "available": True,
        },
        {
            "name": "LogisticRegression",
            "estimator": LogisticRegression(max_iter=1000, random_state=random_state),
            "scale_numeric": True,
            "available": True,
        },
        {
            "name": "DecisionTree",
            "estimator": DecisionTreeClassifier(random_state=random_state),
            "scale_numeric": False,
            "available": True,
        },
        {
            "name": "RandomForest",
            "estimator": RandomForestClassifier(
                n_estimators=150, random_state=random_state, n_jobs=2
            ),
            "scale_numeric": False,
            "available": True,
        },
        {
            "name": "HistGradientBoosting",
            "estimator": HistGradientBoostingClassifier(random_state=random_state),
            "scale_numeric": False,
            "available": True,
        },
    ]

    try:
        from xgboost import XGBClassifier

        specs.append(
            {
                "name": "XGBoost",
                "estimator": XGBClassifier(
                    n_estimators=150,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=random_state,
                    n_jobs=2,
                    tree_method="hist",
                    eval_metric="mlogloss",
                    use_label_encoder=False,
                ),
                "scale_numeric": False,
                "available": True,
            }
        )
    except ImportError as exc:
        specs.append({"name": "XGBoost", "available": False, "error": str(exc)})

    try:
        from lightgbm import LGBMClassifier

        specs.append(
            {
                "name": "LightGBM",
                "estimator": LGBMClassifier(
                    n_estimators=150, random_state=random_state, n_jobs=2, verbose=-1
                ),
                "scale_numeric": False,
                "available": True,
            }
        )
    except ImportError as exc:
        specs.append({"name": "LightGBM", "available": False, "error": str(exc)})

    return specs


def evaluate_candidates(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_set: FeatureSetName = "combined",
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Train and evaluate all Phase 6 candidate models on train -> val."""
    le = get_label_encoder()
    y_train = le.transform(train_df["Load_Type"])
    y_val = le.transform(val_df["Load_Type"])
    label_ids = list(range(len(EXPECTED_CLASSES)))
    label_names = list(le.classes_)

    rows: list[dict[str, Any]] = []
    for spec in get_candidate_specs(random_state=random_state):
        if not spec.get("available", False):
            rows.append(
                {
                    "model": spec["name"],
                    "feature_set": feature_set,
                    "status": f"skipped: {spec.get('error', 'unavailable')}",
                    "train_time_sec": np.nan,
                    "accuracy": np.nan,
                    "macro_precision": np.nan,
                    "macro_recall": np.nan,
                    "macro_f1": np.nan,
                    "weighted_f1": np.nan,
                    "params": "{}",
                }
            )
            continue

        pipe = Pipeline(
            [
                ("prep", build_preprocessor(feature_set=feature_set, scale_numeric=spec["scale_numeric"])),
                ("clf", spec["estimator"]),
            ]
        )
        t0 = time.perf_counter()
        pipe.fit(train_df, y_train)
        train_time = time.perf_counter() - t0
        y_pred = pipe.predict(val_df)

        metrics = compute_classification_metrics(
            y_val, y_pred, labels=label_ids, label_names=label_names
        )
        rows.append(
            {
                "model": spec["name"],
                "feature_set": feature_set,
                "status": "ok",
                "train_time_sec": round(train_time, 3),
                **{k: round(v, 4) for k, v in metrics.items()},
                "params": json.dumps(spec["estimator"].get_params(), default=str),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["macro_f1", "accuracy"], ascending=False, na_position="last"
    ).reset_index(drop=True)


def tune_top_models(
    dev_df: pd.DataFrame,
    n_splits: int = 4,
    n_iter: int = 8,
    random_state: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Run bounded RandomizedSearchCV with TimeSeriesSplit on development data (Jan-Nov) only."""
    le = get_label_encoder()
    y_dev = le.transform(dev_df["Load_Type"])
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Verify strict chronological fold boundaries before tuning
    for train_idx, val_idx in tscv.split(dev_df):
        if not (dev_df.iloc[train_idx]["Date_Time"].max() < dev_df.iloc[val_idx]["Date_Time"].min()):
            raise AssertionError("TimeSeriesSplit fold boundary overlap detected.")

    search_configs: list[dict[str, Any]] = [
        {
            "name": "RandomForest",
            "feature_set": "combined",
            "estimator": RandomForestClassifier(random_state=random_state, n_jobs=1),
            "param_distributions": {
                "clf__n_estimators": [100, 150, 250],
                "clf__max_depth": [10, 15, 25, None],
                "clf__min_samples_leaf": [1, 2, 5],
                "clf__class_weight": [None, "balanced", "balanced_subsample"],
            },
        },
        {
            "name": "HistGradientBoosting",
            "feature_set": "combined",
            "estimator": HistGradientBoostingClassifier(random_state=random_state),
            "param_distributions": {
                "clf__max_iter": [100, 200, 300],
                "clf__max_depth": [4, 6, 10, None],
                "clf__learning_rate": [0.05, 0.1, 0.15],
                "clf__min_samples_leaf": [20, 50, 100],
                "clf__class_weight": [None, "balanced"],
            },
        },
        {
            "name": "RandomForest_measurement_only",
            "feature_set": "measurement_only",
            "estimator": RandomForestClassifier(random_state=random_state, n_jobs=1),
            "param_distributions": {
                "clf__n_estimators": [100, 150, 250],
                "clf__max_depth": [10, 15, 25, None],
                "clf__min_samples_leaf": [1, 2, 5],
                "clf__class_weight": [None, "balanced"],
            },
        },
    ]

    try:
        from xgboost import XGBClassifier

        search_configs.append(
            {
                "name": "XGBoost",
                "feature_set": "combined",
                "estimator": XGBClassifier(
                    random_state=random_state,
                    n_jobs=1,
                    tree_method="hist",
                    eval_metric="mlogloss",
                    use_label_encoder=False,
                ),
                "param_distributions": {
                    "clf__n_estimators": [100, 150, 250],
                    "clf__max_depth": [4, 6, 8],
                    "clf__learning_rate": [0.05, 0.1, 0.15],
                    "clf__subsample": [0.8, 1.0],
                    "clf__colsample_bytree": [0.8, 1.0],
                },
            }
        )
    except ImportError:
        pass

    tuning_records: list[dict[str, Any]] = []
    for cfg in search_configs:
        print(f"Tuning {cfg['name']} ({cfg['feature_set']})...", flush=True)
        pipe = Pipeline(
            [
                ("prep", build_preprocessor(feature_set=cfg["feature_set"], scale_numeric=False)),
                ("clf", cfg["estimator"]),
            ]
        )
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=cfg["param_distributions"],
            n_iter=n_iter,
            scoring="f1_macro",
            cv=tscv,
            random_state=random_state,
            n_jobs=1,
            refit=False,
        )
        t0 = time.perf_counter()
        search.fit(dev_df, y_dev)
        elapsed = time.perf_counter() - t0

        tuning_records.append(
            {
                "name": cfg["name"],
                "feature_set": cfg["feature_set"],
                "best_cv_macro_f1": round(float(search.best_score_), 4),
                "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
                "search_time_sec": round(elapsed, 3),
            }
        )

    tuning_records.sort(key=lambda r: r["best_cv_macro_f1"], reverse=True)
    return {
        "cv_strategy": f"TimeSeriesSplit(n_splits={n_splits})",
        "primary_metric": "f1_macro",
        "n_iter_per_model": n_iter,
        "random_seed": random_state,
        "dev_rows": int(len(dev_df)),
        "winner": tuning_records[0],
        "candidates": tuning_records,
    }


def main() -> None:
    """Run Phase 6 candidate comparison and Phase 7 chronological tuning."""
    project_root = Path(__file__).resolve().parents[2]
    raw_df = load_raw_data(project_root / "data" / "load_data.csv")
    cleaned_df, _ = clean_dataset(raw_df)
    train_df, val_df, _, _ = chronological_split(cleaned_df)
    dev_df = pd.concat([train_df, val_df], ignore_index=True)

    comparison_df = evaluate_candidates(train_df, val_df, feature_set="combined")
    comp_path = project_root / "outputs" / "model_comparison.csv"
    comp_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comp_path, index=False)
    print("Phase 6 Candidate Comparison:")
    print(comparison_df[["model", "status", "train_time_sec", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]].to_string(index=False))

    tuning_results = tune_top_models(dev_df, n_splits=3, n_iter=4)
    tuning_path = project_root / "outputs" / "tuning_results.json"
    tuning_path.write_text(json.dumps(tuning_results, indent=2), encoding="utf-8")
    print("\nPhase 7 Tuning Winner:")
    print(json.dumps(tuning_results["winner"], indent=2))


if __name__ == "__main__":
    main()
