# -*- coding: utf-8 -*-
"""
Train a binary classifier (XGBoost or LightGBM) for bike/dock stockout within 30 minutes,
using a time-based train/validation split and MLflow autologging.
Everything is commented in English for clarity.
"""

import argparse
import json
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

# Metrics & plots
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix

# Tree models
import xgboost as xgb
import lightgbm as lgb

# Tracking
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

# Visualization for artifacts
import matplotlib.pyplot as plt
import seaborn as sns

# AWS Athena connector
from pyathena import connect

# Project schema (feature list, labels, validation helper)
from src.features.schema import FEATURE_COLUMNS, LABEL_COLUMNS, validate_feature_df


# ------------------------
# Config dataclasses
# ------------------------
@dataclass
class DataConfig:
    city: str
    start: str  # 'YYYY-MM-DD HH:MM' UTC
    end: str    # 'YYYY-MM-DD HH:MM' UTC
    athena_database: str
    athena_workgroup: str = "primary"
    athena_output: Optional[str] = None
    region: str = "ca-central-1"


@dataclass
class TrainConfig:
    label: str = "y_stockout_bikes_30"      # or "y_stockout_docks_30"
    model_type: str = "xgboost"             # "xgboost" or "lightgbm"
    valid_ratio: float = 0.2                # later time slice used for validation
    gap_minutes: int = 60                   # anti-leakage time gap between train/valid
    random_state: int = 42
    beta: float = 2.0                       # F-beta (β=2 emphasizes recall)
    experiment: str = "bikeshare-step4"
    # Keep this name EXACTLY as used below to avoid keyword mismatch
    use_sm_experiments_autolog: bool = False
    class_weight: Optional[float] = None    # e.g., positive class weight for imbalance
    xgb_params: dict = None                 # baseline XGBoost params
    lgb_params: dict = None                 # baseline LightGBM params


# ------------------------
# Athena helpers
# ------------------------
def athena_conn(region: str, workgroup: str, s3_staging_dir: Optional[str], schema_name: str):
    """
    Create a PyAthena connection. If s3_staging_dir is provided, use it for query results;
    otherwise rely on workgroup settings.
    """
    if s3_staging_dir:
        return connect(region_name=region, s3_staging_dir=s3_staging_dir, work_group=workgroup, schema_name=schema_name)
    return connect(region_name=region, work_group=workgroup, schema_name=schema_name)


def load_features_offline(cnx, city: str, start_ts: str, end_ts: str, db: str) -> pd.DataFrame:
    """
    Read a time window from the Athena external table features_offline (Step 3 output).
    """
    sql = f"""
    SELECT *
    FROM {db}.features_offline
    WHERE city = '{city}'
      AND parse_datetime(dt, 'yyyy-MM-dd-HH-mm')
          BETWEEN TIMESTAMP '{start_ts}:00' AND TIMESTAMP '{end_ts}:00'
    """
    return pd.read_sql(sql, cnx)


# ------------------------
# Time-based split
# ------------------------
def temporal_split(df: pd.DataFrame, valid_ratio: float, gap_minutes: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into (train, valid) by time. The latest fraction goes to valid.
    A gap is held out to reduce temporal leakage.
    """
    # Parse dt as timezone-aware timestamps for ordering
    ts = pd.to_datetime(df["dt"], format="%Y-%m-%d-%H-%M", utc=True)
    df = df.assign(ts=ts).sort_values("ts")

    # Unique time stamps define the split
    times = df["ts"].unique()
    n = len(times)
    if n < 3:
        raise ValueError("Not enough time points for a temporal split. Expand your time range.")

    split_idx = int(np.floor(n * (1.0 - valid_ratio)))
    split_idx = min(max(split_idx, 1), n - 1)  # keep inside (0, n)

    train_end_time = times[split_idx - 1]
    valid_start_time = times[split_idx]

    gap = pd.Timedelta(minutes=gap_minutes)
    train_mask = df["ts"] <= train_end_time
    valid_mask = df["ts"] >= (valid_start_time + gap)

    train_df = df.loc[train_mask].copy()
    valid_df = df.loc[valid_mask].copy()

    # If the gap makes valid empty, relax gap
    if valid_df.empty:
        valid_df = df[df["ts"] >= valid_start_time].copy()

    return train_df, valid_df


# ------------------------
# Evaluation helpers
# ------------------------
def pick_threshold_fbeta(y_true: np.ndarray, y_prob: np.ndarray, beta: float):
    """
    Grid-search thresholds in [0.01, 0.99] to maximize F-beta.
    Return (best_threshold, metrics_at_best).
    """
    best_t = 0.5
    best_score = -1.0
    best = {}

    pr_auc = average_precision_score(y_true, y_prob)

    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision == 0 and recall == 0:
            fbeta = 0.0
        else:
            b2 = beta * beta
            denom = (b2 * precision + recall)
            fbeta = (1 + b2) * (precision * recall) / denom if denom > 0 else 0.0

        if fbeta > best_score:
            best_score = fbeta
            best_t = float(t)
            best = {
                "precision": float(precision),
                "recall": float(recall),
                "fbeta": float(fbeta),
                "threshold": float(t),
                "pr_auc": float(pr_auc),
            }

    return best_t, best


def log_curves_and_confusion(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, prefix: str = "val"):
    """
    Log PR curve + confusion matrix (PNG) to MLflow.
    """
    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.step(recall, precision, where="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({prefix})")
    pr_path = f"{prefix}_pr_curve.png"
    plt.savefig(pr_path, bbox_inches="tight", dpi=150)
    plt.close()
    mlflow.log_artifact(pr_path)

    # Confusion matrix @ threshold
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
    plt.title(f"Confusion Matrix @ t={threshold:.2f} ({prefix})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = f"{prefix}_confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight", dpi=150)
    plt.close()
    mlflow.log_artifact(cm_path)


def log_feature_importance(model, feature_names: List[str]):
    """
    Save feature importance as CSV and PNG.
    Supports XGBoost (gain) and LightGBM (split counts).
    """
    importances = None

    # Try XGBoost gain
    try:
        booster = model.get_booster() if hasattr(model, "get_booster") else None
        if booster is not None:
            score = booster.get_score(importance_type="gain")  # dict: feat -> gain
            importances = pd.DataFrame([(k, v) for k, v in score.items()], columns=["feature", "importance"])
    except Exception:
        pass

    # Fallback to LightGBM split counts
    if importances is None or importances.empty:
        try:
            arr = getattr(model, "feature_importances_", None)
            if arr is not None:
                importances = pd.DataFrame({"feature": feature_names, "importance": arr})
        except Exception:
            pass

    if importances is None or importances.empty:
        return  # nothing to log

    importances = importances.sort_values("importance", ascending=False)

    csv_path = "feature_importance.csv"
    importances.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)

    top = importances.head(25)
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance"])
    plt.gca().invert_yaxis()
    plt.title("Top Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    png_path = "feature_importance.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    mlflow.log_artifact(png_path)


# ------------------------
# Main training
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    # Data slice
    parser.add_argument("--city", required=True, help="City partition, e.g., nyc")
    parser.add_argument("--start", required=True, help="UTC start 'YYYY-MM-DD HH:MM'")
    parser.add_argument("--end", required=True, help="UTC end 'YYYY-MM-DD HH:MM'")
    parser.add_argument("--database", default="mlops_bikeshare", help="Athena database name")
    parser.add_argument("--workgroup", default="primary", help="Athena workgroup")
    parser.add_argument("--athena-output", default=None, help="Optional S3 path for Athena staging")
    parser.add_argument("--region", default="ca-central-1", help="AWS region")

    # Training options
    parser.add_argument("--label", default="y_stockout_bikes_30",
                        choices=["y_stockout_bikes_30", "y_stockout_docks_30"])
    parser.add_argument("--model-type", default="xgboost", choices=["xgboost", "lightgbm"])
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--gap-minutes", type=int, default=60)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--experiment", default="bikeshare-step4")
    parser.add_argument("--use-sm-exp", action="store_true", help="Enable SageMaker Experiments autolog (optional)")

    args = parser.parse_args()

    # Build configs
    dcfg = DataConfig(
        city=args.city,
        start=args.start,
        end=args.end,
        athena_database=args.database,
        athena_workgroup=args.workgroup,
        athena_output=args.athena_output,
        region=args.region,
    )
    tcfg = TrainConfig(
        label=args.label,
        model_type=args.model_type,
        valid_ratio=args.valid_ratio,
        gap_minutes=args.gap_minutes,
        random_state=args.random_state,
        beta=args.beta,
        experiment=args.experiment,
        use_sm_experiments_autolog=args.use_sm_exp,  # match field name above
        xgb_params={
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "random_state": args.random_state,
        },
        lgb_params={
            "n_estimators": 800,
            "num_leaves": 63,
            "max_depth": -1,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "random_state": args.random_state,
        },
    )

    # Load data from Athena
    cnx = athena_conn(
        region=dcfg.region,
        workgroup=dcfg.athena_workgroup,
        s3_staging_dir=dcfg.athena_output,
        schema_name=dcfg.athena_database,
    )
    raw = load_features_offline(cnx, dcfg.city, dcfg.start, dcfg.end, dcfg.athena_database)
    if raw.empty:
        raise RuntimeError("No rows found in the selected window. Expand --start/--end or check partitions.")

    # Quick schema/quality validation (your project helper)
    validate_feature_df(raw)

    # Select features + label; drop rows with missing label
    features = FEATURE_COLUMNS
    label = tcfg.label
    df = raw[["city", "dt", "station_id"] + features + [label]].copy()
    df = df.dropna(subset=[label])

    # Temporal split
    train_df, valid_df = temporal_split(df, tcfg.valid_ratio, tcfg.gap_minutes)

    X_train = train_df[features].astype("float32")
    y_train = train_df[label].astype(int).values
    X_valid = valid_df[features].astype("float32")
    y_valid = valid_df[label].astype(int).values

    # MLflow experiment
    mlflow.set_experiment(tcfg.experiment)

    with mlflow.start_run(run_name=f"{tcfg.model_type}-{label}") as run:
        run_id = run.info.run_id

        # Autologging for selected model family
        if tcfg.model_type == "xgboost":
            mlflow.xgboost.autolog(log_models=True)
            model = xgb.XGBClassifier(**tcfg.xgb_params, n_jobs=0)
        else:
            mlflow.lightgbm.autolog(log_models=True)
            model = lgb.LGBMClassifier(**tcfg.lgb_params)

        # Fit
        model.fit(X_train, y_train)

        # Predict probabilities
        train_prob = model.predict_proba(X_train)[:, 1]
        valid_prob = model.predict_proba(X_valid)[:, 1]

        # Core metrics
        pr_auc_train = float(average_precision_score(y_train, train_prob))
        pr_auc_valid = float(average_precision_score(y_valid, valid_prob))
        gap = abs(pr_auc_train - pr_auc_valid)

        # Threshold by F-beta on valid
        best_t, best_metrics = pick_threshold_fbeta(y_valid, valid_prob, beta=tcfg.beta)

        # Log all metrics at once
        mlflow.log_metrics({
            "pr_auc_train": pr_auc_train,
            "pr_auc_valid": pr_auc_valid,
            "overfit_gap": gap,
            "best_precision": best_metrics["precision"],
            "best_recall": best_metrics["recall"],
            "best_fbeta": best_metrics["fbeta"],
            "beta": tcfg.beta,
            "best_threshold": best_t
        })

        # Artifacts
        log_curves_and_confusion(y_valid, valid_prob, threshold=best_t, prefix="val")
        log_curves_and_confusion(y_train, train_prob, threshold=best_t, prefix="train")
        log_feature_importance(model, features)

        # Save eval summary for model_card.md generation
        eval_blob = {
            "label": label,
            "model_type": tcfg.model_type,
            "features": features,
            "n_train": int(len(X_train)),
            "n_valid": int(len(X_valid)),
            "pr_auc_train": pr_auc_train,
            "pr_auc_valid": pr_auc_valid,
            "overfit_gap": gap,
            "best_threshold": best_t,
            "best_precision": best_metrics["precision"],
            "best_recall": best_metrics["recall"],
            "best_fbeta": best_metrics["fbeta"],
            "beta": tcfg.beta,
            "city": dcfg.city,
            "time_start": dcfg.start,
            "time_end": dcfg.end,
            "run_id": run_id,
        }
        with open("eval_summary.json", "w", encoding="utf-8") as f:
            json.dump(eval_blob, f, indent=2)
        mlflow.log_artifact("eval_summary.json")

        # Explicit log (autolog already logs a model, but this gives a stable path)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Soft overfitting check
        if gap >= 0.10:
            print(f"[WARN] Overfitting check failed: train-valid PR-AUC gap={gap:.3f} ≥ 0.10")

        print(f"[OK] Training done. PR-AUC (train)={pr_auc_train:.3f}, (valid)={pr_auc_valid:.3f}, gap={gap:.3f}")
        print(f"[OK] Best threshold (beta={tcfg.beta}): {best_t:.2f}; Fbeta={best_metrics['fbeta']:.3f}")


if __name__ == "__main__":
    main()
