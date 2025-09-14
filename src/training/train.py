# -*- coding: utf-8 -*-
"""
Train a binary classifier (XGBoost or LightGBM) for bike/dock stockout within 30 minutes,
using a time-based train/validation split and MLflow autologging.
Everything is commented in English for clarity.
"""
import argparse
import os
from dataclasses import dataclass
from typing import List, Optional

import lightgbm as lgb

# Visualization for artifacts
import matplotlib.pyplot as plt

# Tracking
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns

# Tree models
import xgboost as xgb
from mlflow.models.signature import infer_signature

# AWS Athena connector
from pyathena import connect

# Metrics & plots
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

# Project schema (feature list, labels, required base cols, validator)
from src.features.schema import (
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    REQUIRED_BASE,
    validate_feature_df,
)

# -------------------------------------------------------------------
# MLflow tracking setup
# -------------------------------------------------------------------
# If the environment already defines MLFLOW_TRACKING_URI (e.g., to a local UI server or remote),
# we respect it. Otherwise, default to a lightweight local SQLite file-store (Windows-friendly).
if not os.environ.get("MLFLOW_TRACKING_URI"):
    # Stored in repo working dir; works fine with `mlflow ui` or reading artifacts locally.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")


# ------------------------
# Config dataclasses
# ------------------------
@dataclass
class DataConfig:
    city: str
    start: str  # 'YYYY-MM-DD HH:MM' UTC
    end: str  # 'YYYY-MM-DD HH:MM' UTC
    athena_database: str
    athena_workgroup: str = "primary"
    athena_output: Optional[str] = None
    region: str = "ca-central-1"


@dataclass
class TrainConfig:
    label: str = "y_stockout_bikes_30"  # or "y_stockout_docks_30"
    model_type: str = "xgboost"  # "xgboost" or "lightgbm"
    valid_ratio: float = 0.2  # later time slice used for validation
    gap_minutes: int = 60  # anti-leakage time gap between train/valid
    random_state: int = 42
    beta: float = 2.0  # F-beta (β=2 emphasizes recall)
    experiment: str = "bikeshare-step4"
    # Keep this name EXACTLY as used below to avoid keyword mismatch
    use_sm_experiments_autolog: bool = False
    class_weight: Optional[float] = None  # e.g., positive class weight for imbalance
    xgb_params: dict = None  # baseline XGBoost params
    lgb_params: dict = None  # baseline LightGBM params


class ProbWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper so that .predict() returns P(y=1) instead of hard 0/1 labels."""

    def __init__(self, base_model, feature_names):
        # Store trained classifier and the feature list
        self._m = base_model
        self._feat = feature_names

    def predict(self, context, model_input):
        # Ensure DataFrame input with exact feature order
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input, columns=self._feat)
        X = model_input[self._feat].astype("float32")
        # Return probability for positive class
        return self._m.predict_proba(X)[:, 1]


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


# def load_features_offline(cnx, city: str, start_ts: str, end_ts: str, db: str) -> pd.DataFrame:
#     """
#     Read a time window from the Athena external table features_offline (Step 3 output).
#     """
#     sql = f"""
#     SELECT *
#     FROM {db}.features_offline
#     WHERE city = '{city}'
#       AND parse_datetime(dt, 'yyyy-MM-dd-HH-mm')
#           BETWEEN TIMESTAMP '{start_ts}:00' AND TIMESTAMP '{end_ts}:00'
#     """
#     return pd.read_sql(sql, cnx)


def list_unique_dt(cnx, db: str, city: str, start_ts: str, end_ts: str) -> list:
    """
    Fetch only the distinct 'dt' strings in the requested window.
    This is tiny (minutes) compared to full rows.
    """
    sql = f"""
    SELECT DISTINCT dt
    FROM {db}.features_offline
    WHERE city = '{city}'
      AND parse_datetime(dt, 'yyyy-MM-dd-HH-mm')
          BETWEEN TIMESTAMP '{start_ts}:00' AND TIMESTAMP '{end_ts}:00'
    ORDER BY dt
    """
    # This result is small; safe to load
    s = pd.read_sql(sql, cnx)["dt"].tolist()
    return s


def load_slice(cnx, db: str, city: str, features: list, labels: list, dt_cond_sql: str) -> pd.DataFrame:
    """
    Load a train OR valid slice with all schema-required columns:
    REQUIRED_BASE + features + all labels.
    We quote & alias every identifier to keep exact column names.
    """
    # Build the select list in the schema order
    select_cols = list(REQUIRED_BASE) + list(features) + list(labels)

    # Deduplicate while preserving order
    seen = set()
    select_cols = [c for c in select_cols if not (c in seen or seen.add(c))]

    # Quote and alias each identifier: "col" AS "col"
    select_list = ", ".join([f'"{c}" AS "{c}"' for c in select_cols])

    sql = f"""
    SELECT {select_list}
    FROM {db}.features_offline
    WHERE "city" = '{city}' AND ({dt_cond_sql})
    """
    return pd.read_sql(sql, cnx)


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
            denom = b2 * precision + recall
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


def log_curves_and_confusion(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, out_dir_plots: str, prefix: str = "val"
):
    """
    Log PR curve + confusion matrix (PNG) to MLflow.
    """
    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig_pr = plt.figure()
    plt.step(recall, precision, where="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({prefix})")
    mlflow.log_figure(fig_pr, f"{out_dir_plots}/{prefix}_pr_curve.png")  # stored under mlruns/.../artifacts
    plt.close(fig_pr)

    # Confusion matrix @ threshold
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig_cm = plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
    plt.title(f"Confusion Matrix @ t={threshold:.2f} ({prefix})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    mlflow.log_figure(fig_cm, f"{out_dir_plots}/{prefix}_confusion_matrix.png")
    plt.close(fig_cm)


def log_feature_importance(model, feature_names: List[str], out_dir_featimp: str):
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

    csv_blob = importances.to_csv(index=False)
    mlflow.log_text(csv_blob, f"{out_dir_featimp}/feature_importance.csv")

    top = importances.head(25)
    fig = plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance"])
    plt.gca().invert_yaxis()
    plt.title("Top Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    mlflow.log_figure(fig, f"{out_dir_featimp}/feature_importance.png")
    plt.close(fig)


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
    parser.add_argument(
        "--label", default="y_stockout_bikes_30", choices=["y_stockout_bikes_30", "y_stockout_docks_30"]
    )
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

    # --------------------------
    # Unified artifact namespaces
    # --------------------------
    task_id = f"{dcfg.city}_{tcfg.label}_{tcfg.model_type}"  # e.g., nyc_y_stockout_bikes_30_xgboost

    DIR_PLOTS = f"artifacts/{task_id}/plots"  # curves, confusion matrices
    DIR_EVAL = f"artifacts/{task_id}/eval"  # eval json, metrics
    DIR_FEATIMP = f"artifacts/{task_id}/feature_importance"

    MODEL_NAME_BASE = (
        f"{dcfg.city}_{tcfg.label}_{tcfg.model_type}__base"  # e.g., nyc_y_stockout_bikes_30_xgboost__base_sklearn
    )
    MODEL_NAME_PROBA = f"{task_id}__model_proba"  # e.g., nyc_y_stockout_bikes_30_xgboost__model_proba

    # Get the distinct dt’s in the window (lightweight)
    dt_list = list_unique_dt(cnx, dcfg.athena_database, dcfg.city, dcfg.start, dcfg.end)
    if len(dt_list) < 3:
        raise RuntimeError("Not enough time points. Widen --start/--end.")

    # Decide temporal split based on dt_list
    n = len(dt_list)
    split_idx = int(np.floor(n * (1.0 - tcfg.valid_ratio)))
    split_idx = min(max(split_idx, 1), n - 1)

    train_end_dt = dt_list[split_idx - 1]  # inclusive
    valid_start_dt = dt_list[split_idx]  # boundary before applying gap

    # Apply the anti-leakage gap (5-min grid)
    ticks = int(np.ceil(tcfg.gap_minutes / 5.0))
    valid_start_idx = min(split_idx + ticks, n - 1)
    valid_start_dt = dt_list[valid_start_idx]

    features = FEATURE_COLUMNS
    label = tcfg.label

    # TRAIN: dt <= train_end_dt  (and >= requested start)
    train_where = f"\"dt\" >= '{dcfg.start.replace(' ', '-')}' AND \"dt\" <= '{train_end_dt}'"
    train_df = load_slice(cnx, dcfg.athena_database, dcfg.city, features, LABEL_COLUMNS, train_where)
    validate_feature_df(train_df)  # your schema checks
    train_df = train_df.dropna(subset=[label])

    # VALID: dt >= valid_start_dt (and <= requested end)
    valid_where = f"\"dt\" >= '{valid_start_dt}' AND \"dt\" <= '{dcfg.end.replace(' ', '-')}'"
    valid_df = load_slice(cnx, dcfg.athena_database, dcfg.city, features, LABEL_COLUMNS, valid_where)
    validate_feature_df(valid_df)
    valid_df = valid_df.dropna(subset=[label])

    X_train = train_df[features].astype("float32")
    y_train = train_df[label].astype(int).values
    X_valid = valid_df[features].astype("float32")
    y_valid = valid_df[label].astype(int).values

    # MLflow experiment
    mlflow.set_experiment(tcfg.experiment)

    mlflow.autolog(disable=True)

    with mlflow.start_run(run_name=f"{tcfg.model_type}-{label}") as run:
        run_id = run.info.run_id
        print("RUN_ID:", run.info.run_id)

        # Autologging for selected model family
        if tcfg.model_type == "xgboost":
            # mlflow.xgboost.autolog(log_models=True)
            model = xgb.XGBClassifier(**tcfg.xgb_params, n_jobs=0)
        else:
            # mlflow.lightgbm.autolog(log_models=True)
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
        mlflow.log_metrics(
            {
                "pr_auc_train": pr_auc_train,
                "pr_auc_valid": pr_auc_valid,
                "overfit_gap": gap,
                "best_precision": best_metrics["precision"],
                "best_recall": best_metrics["recall"],
                "best_fbeta": best_metrics["fbeta"],
                "beta": tcfg.beta,
                "best_threshold": best_t,
            }
        )

        # Artifacts
        log_curves_and_confusion(
            y_valid,
            valid_prob,
            threshold=best_t,
            out_dir_plots=DIR_PLOTS,
            prefix="val",
        )
        log_curves_and_confusion(y_train, train_prob, threshold=best_t, out_dir_plots=DIR_PLOTS, prefix="train")
        log_feature_importance(model, features, out_dir_featimp=DIR_FEATIMP)

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
        # with open("eval_summary.json", "w", encoding="utf-8") as f:
        #     json.dump(eval_blob, f, indent=2)
        mlflow.log_dict(eval_blob, f"{DIR_EVAL}/eval_summary.json")

        # Explicit log (autolog already logs a model, but this gives a stable path)
        mlflow.sklearn.log_model(model, name=MODEL_NAME_BASE)

        # Build a tiny sample input for model signature
        sample_X = pd.DataFrame({f: pd.Series([0.0], dtype="float32") for f in features})[features]

        signature = infer_signature(sample_X, pd.Series([0.5], dtype="float64"))

        # Log wrapped model as a separate artifact path in MLflow
        mlflow.pyfunc.log_model(
            name=MODEL_NAME_PROBA,
            python_model=ProbWrapper(model, features),
            signature=signature,
            input_example=sample_X,
        )

        print(f"[OK] Logged probability-serving model as MLflow model '{MODEL_NAME_PROBA}'")

        # Soft overfitting check
        if gap >= 0.10:
            print(f"[WARN] Overfitting check failed: train-valid PR-AUC gap={gap:.3f} ≥ 0.10")

        print(f"[OK] Training done. PR-AUC (train)={pr_auc_train:.3f}, (valid)={pr_auc_valid:.3f}, gap={gap:.3f}")
        print(f"[OK] Best threshold (beta={tcfg.beta}): {best_t:.2f}; Fbeta={best_metrics['fbeta']:.3f}")


if __name__ == "__main__":
    main()
