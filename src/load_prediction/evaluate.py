"""Phases 8-9: Final evaluation on held-out December test set, explainability, and error analysis."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "outputs" / ".mplconfig"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from src.load_prediction.data import (
    EXPECTED_CLASSES,
    chronological_split,
    clean_dataset,
    load_raw_data,
)
from src.load_prediction.features import build_preprocessor
from src.load_prediction.train import (
    RANDOM_SEED,
    compute_classification_metrics,
    get_label_encoder,
)

sns.set_theme(style="whitegrid", context="notebook")


def build_model_from_config(winner_cfg: dict[str, Any], random_state: int = RANDOM_SEED) -> Pipeline:
    """Instantiate the locked winning Pipeline from tuning_results.json."""
    name = winner_cfg["name"]
    feature_set = winner_cfg.get("feature_set", "combined")
    params = dict(winner_cfg.get("best_params", {}))

    if name.startswith("RandomForest"):
        clf = RandomForestClassifier(random_state=random_state, n_jobs=1, **params)
    elif name == "HistGradientBoosting":
        clf = HistGradientBoostingClassifier(random_state=random_state, **params)
    elif name == "XGBoost":
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            random_state=random_state,
            n_jobs=1,
            tree_method="hist",
            eval_metric="mlogloss",
            use_label_encoder=False,
            **params,
        )
    else:
        raise ValueError(f"Unsupported winner model name: {name!r}")

    return Pipeline(
        [
            ("prep", build_preprocessor(feature_set=feature_set, scale_numeric=False)),
            ("clf", clf),
        ]
    )


def evaluate_final_model(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    winner_cfg: dict[str, Any],
    outputs_dir: str | Path,
    models_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any], Pipeline]:
    """Refit locked pipeline on Jan-Nov dev data and evaluate once on Dec test data."""
    out_dir = Path(outputs_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    mod_dir = Path(models_dir)
    mod_dir.mkdir(parents=True, exist_ok=True)

    le = get_label_encoder()
    y_dev = le.transform(dev_df["Load_Type"])
    y_test = le.transform(test_df["Load_Type"])
    label_ids = list(range(len(EXPECTED_CLASSES)))
    label_names = list(le.classes_)

    pipe = build_model_from_config(winner_cfg)
    pipe.fit(dev_df, y_dev)

    y_pred = pipe.predict(test_df)
    y_prob = pipe.predict_proba(test_df) if hasattr(pipe, "predict_proba") else None

    pred_df = pd.DataFrame(
        {
            "Date_Time": test_df["Date_Time"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "actual_load_type": le.inverse_transform(y_test),
            "predicted_load_type": le.inverse_transform(y_pred),
        }
    )
    if y_prob is not None:
        for idx, cls_name in enumerate(label_names):
            pred_df[f"prob_{cls_name}"] = np.round(y_prob[:, idx], 6)

    pred_path = out_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    summary_metrics = compute_classification_metrics(
        y_test, y_pred, labels=label_ids, label_names=label_names
    )
    cls_report = classification_report(
        y_test,
        y_pred,
        labels=label_ids,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=label_ids)

    metrics_payload: dict[str, Any] = {
        "test_sample_count": int(len(test_df)),
        "train_dev_sample_count": int(len(dev_df)),
        "model_configuration": winner_cfg,
        "summary_metrics": {k: round(v, 4) for k, v in summary_metrics.items()},
        "per_class_report": cls_report,
        "confusion_matrix": {
            "labels": label_names,
            "matrix": cm.tolist(),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # Save confusion matrix figure
    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_title("Final Test Set (December) Confusion Matrix")
    ax.set_xlabel("Predicted Load Type")
    ax.set_ylabel("Actual Load Type")
    fig.tight_layout()
    fig.savefig(fig_dir / "08_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Save fitted pipeline + metadata
    joblib.dump(pipe, mod_dir / "best_model.joblib")
    model_meta = {
        "model_name": winner_cfg["name"],
        "feature_set": winner_cfg.get("feature_set", "combined"),
        "best_params": winner_cfg.get("best_params", {}),
        "classes": label_names,
        "train_rows": int(len(dev_df)),
        "security_note": "Never load untrusted .joblib files; joblib uses pickle deserialization.",
    }
    (mod_dir / "model_metadata.json").write_text(json.dumps(model_meta, indent=2), encoding="utf-8")

    return pred_df, metrics_payload, pipe


def run_explainability_and_error_analysis(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    winner_cfg: dict[str, Any],
    outputs_dir: str | Path,
) -> dict[str, Any]:
    """Compute permutation importance on validation data and descriptive error analysis on test."""
    out_dir = Path(outputs_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    le = get_label_encoder()
    y_train = le.transform(train_df["Load_Type"])
    y_val = le.transform(val_df["Load_Type"])

    # Fit on train_df (Jan-Oct) and compute permutation importance on val_df (Nov)
    dev_pipe = build_model_from_config(winner_cfg)
    dev_pipe.fit(train_df, y_train)

    input_cols = [c for c in val_df.columns if c != "Load_Type"]
    perm = permutation_importance(
        dev_pipe,
        val_df[input_cols],
        y_val,
        scoring="f1_macro",
        n_repeats=5,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    perm_df = (
        pd.DataFrame(
            {
                "feature": input_cols,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    top_perm = perm_df.head(10).iloc[::-1]
    ax.barh(top_perm["feature"], top_perm["importance_mean"], xerr=top_perm["importance_std"], color="#4c72b0")
    ax.set_title("Permutation Importance on Validation Set (Macro F1 Drop)")
    ax.set_xlabel("Mean Macro F1 Decrease")
    fig.tight_layout()
    fig.savefig(fig_dir / "09_permutation_importance.png", dpi=150)
    plt.close(fig)

    # Descriptive error analysis on December test predictions (no retuning)
    err_df = test_df.copy().reset_index(drop=True)
    err_df["actual"] = pred_df["actual_load_type"]
    err_df["predicted"] = pred_df["predicted_load_type"]
    err_df["is_error"] = (err_df["actual"] != err_df["predicted"]).astype(int)
    err_df["hour"] = err_df["Date_Time"].dt.hour
    err_df["is_weekend"] = (err_df["Date_Time"].dt.dayofweek >= 5).astype(int)

    mistakes = err_df[err_df["is_error"] == 1]
    confusion_pairs = (
        mistakes.groupby(["actual", "predicted"]).size().sort_values(ascending=False).to_dict()
    )
    confusion_pairs_str = {f"{a} -> {p}": int(cnt) for (a, p), cnt in confusion_pairs.items()}

    hourly_err = err_df.groupby("hour")["is_error"].mean()
    weekend_err = err_df.groupby("is_weekend")["is_error"].mean().to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    axes[0].plot(hourly_err.index, hourly_err.values * 100, marker="o", color="#c44e52")
    axes[0].set_title("December Test Error Rate by Hour of Day")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Error Rate (%)")
    axes[0].set_xticks(range(0, 24, 2))

    if confusion_pairs_str:
        pairs_s = pd.Series(confusion_pairs_str).head(6).iloc[::-1]
        pairs_s.plot(kind="barh", ax=axes[1], color="#dd8452")
    axes[1].set_title("Top Misclassification Pairs (December Test)")
    axes[1].set_xlabel("Error Count")
    fig.tight_layout()
    fig.savefig(fig_dir / "10_error_analysis.png", dpi=150)
    plt.close(fig)

    explain_payload = {
        "permutation_importance_val": [
            {
                "feature": row["feature"],
                "importance_mean": round(float(row["importance_mean"]), 5),
                "importance_std": round(float(row["importance_std"]), 5),
            }
            for _, row in perm_df.iterrows()
        ],
        "test_error_summary": {
            "total_test_rows": int(len(err_df)),
            "total_errors": int(len(mistakes)),
            "overall_error_rate": round(float(err_df["is_error"].mean()), 4),
            "weekday_error_rate": round(float(weekend_err.get(0, 0.0)), 4),
            "weekend_error_rate": round(float(weekend_err.get(1, 0.0)), 4),
            "confusion_pairs": confusion_pairs_str,
        },
    }
    (out_dir / "explainability.json").write_text(json.dumps(explain_payload, indent=2), encoding="utf-8")
    return explain_payload


def main() -> None:
    """Run Phase 8 final evaluation and Phase 9 explainability & error analysis."""
    project_root = Path(__file__).resolve().parents[2]
    outputs_dir = project_root / "outputs"
    models_dir = project_root / "models"

    tuning_path = outputs_dir / "tuning_results.json"
    tuning_data = json.loads(tuning_path.read_text(encoding="utf-8"))
    winner_cfg = tuning_data["winner"]

    raw_df = load_raw_data(project_root / "data" / "load_data.csv")
    cleaned_df, _ = clean_dataset(raw_df)
    train_df, val_df, test_df, _ = chronological_split(cleaned_df)
    dev_df = pd.concat([train_df, val_df], ignore_index=True)

    pred_df, metrics_payload, _ = evaluate_final_model(
        dev_df, test_df, winner_cfg, outputs_dir, models_dir
    )
    print("Phase 8 Final Test Metrics (December):")
    print(json.dumps(metrics_payload["summary_metrics"], indent=2))

    explain_payload = run_explainability_and_error_analysis(
        train_df, val_df, test_df, pred_df, winner_cfg, outputs_dir
    )
    print("\nPhase 9 Explainability & Error Summary:")
    print(json.dumps(explain_payload["test_error_summary"], indent=2))


if __name__ == "__main__":
    main()
