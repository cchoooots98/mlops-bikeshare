import argparse
import inspect
import json
import math
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import lightgbm as lgb
import matplotlib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

from src.features.postgres_store import (
    PostgresFeatureConfig,
    create_pg_engine,
    list_unique_dt_postgres,
    load_training_slice,
    validate_identifier,
)
from src.features.postgres_store import (
    build_feature_select_query as build_postgres_feature_select_query,
)
from src.features.schema import FEATURE_COLUMNS, LABEL_COLUMNS, REQUIRED_BASE
from src.mlflow_pyfunc_model import PositiveClassProbabilityModel
from src.model_package import (
    ARTIFACTS_DIRNAME,
    DEFAULT_PACKAGE_ROOT,
    MODEL_DIRNAME,
    build_model_card_text,
    build_package_manifest,
    default_package_root_for_target,
    ensure_package_dir,
    write_json_file,
    write_package_manifest,
)
from src.model_target import parse_bool_value, target_spec_from_predict_bikes

DEFAULT_LOCAL_MLFLOW_TRACKING_URI = "sqlite:///model_dir/mlflow.db"

if not os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(DEFAULT_LOCAL_MLFLOW_TRACKING_URI)

DT_FORMAT = "%Y-%m-%d-%H-%M"
INPUT_TS_FORMAT = "%Y-%m-%d %H:%M"
FEATURE_CONTRACT_VERSION = "v1_dim_weather_aligned"
TRAINING_RESULT_PREFIX = "TRAINING_RESULT_JSON::"


@dataclass(frozen=True)
class DataConfig:
    city: str
    start: str
    end: str
    pg_host: str
    pg_port: int
    pg_db: str
    pg_user: str
    pg_password: str
    pg_schema: str = "analytics"
    feature_table: str = "feat_station_snapshot_5min"


@dataclass(frozen=True)
class TemporalSplit:
    train_end_dt: str
    valid_start_dt: str
    split_index: int
    valid_start_index: int
    gap_ticks: int


@dataclass
class TrainConfig:
    predict_bikes: bool = True
    model_type: str = "xgboost"
    valid_ratio: float = 0.2
    gap_minutes: int = 60
    random_state: int = 42
    beta: float = 2.0
    experiment: str = "bikeshare-step4"
    run_reason: str = "manual"
    package_root: str = str(DEFAULT_PACKAGE_ROOT)
    xgb_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "n_jobs": 0,
        }
    )
    lgb_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 800,
            "num_leaves": 63,
            "max_depth": -1,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "verbosity": -1,
        }
    )


def normalize_timestamp(ts: str) -> str:
    return datetime.strptime(ts, INPUT_TS_FORMAT).strftime(DT_FORMAT)


def build_feature_source_name(data_config: DataConfig) -> str:
    return f"{validate_identifier(data_config.pg_schema)}.{validate_identifier(data_config.feature_table)}"


def build_model_name(city: str, label: str, model_type: str) -> str:
    return f"{city}_{label}_{model_type}"


def build_feature_select_query(schema_name: str, table_name: str, select_columns: Sequence[str]) -> str:
    return build_postgres_feature_select_query(
        schema_name,
        table_name,
        select_columns,
        "city = :city AND dt >= :start_dt AND dt <= :end_dt",
        "ORDER BY dt, station_id",
    )


def dedupe_columns(columns: Iterable[str]) -> list[str]:
    seen = set()
    ordered = []
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        ordered.append(column)
    return ordered


def build_pg_config(data_config: DataConfig) -> PostgresFeatureConfig:
    return PostgresFeatureConfig(
        pg_host=data_config.pg_host,
        pg_port=data_config.pg_port,
        pg_db=data_config.pg_db,
        pg_user=data_config.pg_user,
        pg_password=data_config.pg_password,
        pg_schema=data_config.pg_schema,
        training_table=data_config.feature_table,
    )


def load_slice_postgres(
    engine,
    data_config: DataConfig,
    select_columns: Sequence[str],
    start_dt: str,
    end_dt: str,
) -> pd.DataFrame:
    return load_training_slice(
        engine,
        build_pg_config(data_config),
        data_config.city,
        start_dt,
        end_dt,
        select_columns=select_columns,
    )


def compute_temporal_split(dt_list: Sequence[str], valid_ratio: float, gap_minutes: int) -> TemporalSplit:
    if len(dt_list) < 3:
        raise RuntimeError("not enough time points for temporal split; widen --start/--end")
    if not 0 < valid_ratio < 1:
        raise ValueError(f"valid_ratio must be between 0 and 1, got {valid_ratio}")
    if gap_minutes < 0:
        raise ValueError(f"gap_minutes must be non-negative, got {gap_minutes}")

    n_items = len(dt_list)
    split_index = int(math.floor(n_items * (1.0 - valid_ratio)))
    split_index = min(max(split_index, 1), n_items - 1)
    gap_ticks = int(math.ceil(gap_minutes / 5.0))
    valid_start_index = min(split_index + gap_ticks, n_items - 1)
    return TemporalSplit(
        train_end_dt=dt_list[split_index - 1],
        valid_start_dt=dt_list[valid_start_index],
        split_index=split_index,
        valid_start_index=valid_start_index,
        gap_ticks=gap_ticks,
    )


def pick_threshold_fbeta(y_true: np.ndarray, y_prob: np.ndarray, beta: float) -> tuple[float, dict]:
    best_threshold = 0.5
    best_score = -1.0
    best_metrics = {}
    pr_auc = average_precision_score(y_true, y_prob)

    for threshold in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision == 0 and recall == 0:
            fbeta = 0.0
        else:
            beta_sq = beta * beta
            denominator = beta_sq * precision + recall
            fbeta = (1 + beta_sq) * (precision * recall) / denominator if denominator > 0 else 0.0
        if fbeta > best_score:
            best_score = fbeta
            best_threshold = float(threshold)
            best_metrics = {
                "precision": float(precision),
                "recall": float(recall),
                "fbeta": float(fbeta),
                "threshold": float(threshold),
                "pr_auc": float(pr_auc),
            }

    return best_threshold, best_metrics


def log_curves_and_confusion(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    out_dir_plots: str,
    prefix: str,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig_pr = plt.figure()
    plt.step(recall, precision, where="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({prefix})")
    mlflow.log_figure(fig_pr, f"{out_dir_plots}/{prefix}_pr_curve.png")
    plt.close(fig_pr)

    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig_cm = plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
    plt.title(f"Confusion Matrix @ t={threshold:.2f} ({prefix})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    mlflow.log_figure(fig_cm, f"{out_dir_plots}/{prefix}_confusion_matrix.png")
    plt.close(fig_cm)


def log_feature_importance(model, feature_names: Sequence[str], artifact_dir: str) -> None:
    importances = None
    try:
        booster = model.get_booster() if hasattr(model, "get_booster") else None
        if booster is not None:
            score = booster.get_score(importance_type="gain")
            importances = pd.DataFrame(score.items(), columns=["feature", "importance"])
    except Exception:
        importances = None

    if importances is None or importances.empty:
        raw_importance = getattr(model, "feature_importances_", None)
        if raw_importance is not None:
            importances = pd.DataFrame({"feature": feature_names, "importance": raw_importance})

    if importances is None or importances.empty:
        return

    importances = importances.sort_values("importance", ascending=False)
    mlflow.log_text(importances.to_csv(index=False), f"{artifact_dir}/feature_importance.csv")

    top = importances.head(25)
    fig = plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance"])
    plt.gca().invert_yaxis()
    plt.title("Top Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    mlflow.log_figure(fig, f"{artifact_dir}/feature_importance.png")
    plt.close(fig)


def build_sample_input_frame(feature_names: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series([0.0], dtype="float64") for column in feature_names})[list(feature_names)]


def build_classifier(train_config: TrainConfig):
    if train_config.model_type == "xgboost":
        params = dict(train_config.xgb_params)
        params["random_state"] = train_config.random_state
        return xgb.XGBClassifier(**params)
    params = dict(train_config.lgb_params)
    params["random_state"] = train_config.random_state
    return lgb.LGBMClassifier(**params)


def _ensure_binary_classes(y_values: np.ndarray, split_name: str, label_column: str) -> None:
    unique_values = sorted(set(int(value) for value in y_values.tolist()))
    if unique_values != [0, 1]:
        raise RuntimeError(
            f"{split_name} slice for {label_column} must contain both classes [0, 1]; got {unique_values}"
        )


def build_pyfunc_model_kwargs(
    *,
    model,
    model_type: str,
    feature_names: Sequence[str],
    signature,
    input_example: pd.DataFrame,
) -> dict:
    return {
        "python_model": PositiveClassProbabilityModel(
            model=model,
            model_type=model_type,
            feature_names=list(feature_names),
        ),
        "signature": signature,
        "input_example": input_example,
    }


def _save_or_log_pyfunc_probability_model(
    *,
    model,
    model_type: str,
    feature_names: Sequence[str],
    signature,
    input_example: pd.DataFrame,
    save_path: str | None = None,
    log_name: str | None = None,
) -> None:
    if bool(save_path) == bool(log_name):
        raise ValueError("provide exactly one of save_path or log_name")

    common_kwargs = build_pyfunc_model_kwargs(
        model=model,
        model_type=model_type,
        feature_names=feature_names,
        signature=signature,
        input_example=input_example,
    )
    if save_path:
        mlflow.pyfunc.save_model(path=save_path, **common_kwargs)
    else:
        log_model_params = inspect.signature(mlflow.pyfunc.log_model).parameters
        if "name" in log_model_params:
            mlflow.pyfunc.log_model(name=log_name, **common_kwargs)
        else:
            mlflow.pyfunc.log_model(artifact_path=log_name, **common_kwargs)


def _save_local_package(
    *,
    eval_summary: dict,
    run,
    model,
    package_root: str,
) -> tuple[str, str]:
    package_dir = ensure_package_dir(eval_summary["model_name"], run.info.run_id, root_dir=package_root)
    model_dir = package_dir / MODEL_DIRNAME
    artifacts_dir = package_dir / ARTIFACTS_DIRNAME

    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    sample_x = build_sample_input_frame(FEATURE_COLUMNS)
    signature = infer_signature(sample_x, pd.Series([0.5], dtype="float64"))
    _save_or_log_pyfunc_probability_model(
        model=model,
        model_type=eval_summary["model_type"],
        feature_names=FEATURE_COLUMNS,
        signature=signature,
        input_example=sample_x,
        save_path=str(model_dir),
    )

    manifest = build_package_manifest(eval_summary, package_dir)
    manifest_path = write_package_manifest(package_dir, manifest)
    write_json_file(artifacts_dir / "eval_summary.json", eval_summary)
    (artifacts_dir / "model_card.md").write_text(build_model_card_text(eval_summary), encoding="utf-8")
    return str(package_dir.resolve()), manifest_path


def build_eval_summary(
    *,
    run_id: str,
    train_config: TrainConfig,
    data_config: DataConfig,
    target_spec,
    feature_source: str,
    split: TemporalSplit,
    n_train: int,
    n_valid: int,
    train_rows_before_filter: int,
    valid_rows_before_filter: int,
    pr_auc_train: float,
    pr_auc_valid: float,
    overfit_gap: float,
    best_threshold: float,
    best_metrics: dict,
) -> dict:
    return {
        "run_id": run_id,
        "experiment": train_config.experiment,
        "model_name": build_model_name(data_config.city, target_spec.label_column, train_config.model_type),
        "model_type": train_config.model_type,
        "predict_bikes": target_spec.predict_bikes,
        "target_name": target_spec.target_name,
        "label": target_spec.label_column,
        "label_column": target_spec.label_column,
        "paired_target": target_spec.paired_target_column,
        "paired_target_column": target_spec.paired_target_column,
        "score_column": target_spec.score_column,
        "score_bin_column": target_spec.score_bin_column,
        "actual_t30_column": target_spec.actual_t30_column,
        "city": data_config.city,
        "time_start": data_config.start,
        "time_end": data_config.end,
        "train_end_dt": split.train_end_dt,
        "valid_start_dt": split.valid_start_dt,
        "run_reason": train_config.run_reason,
        "feature_source": feature_source,
        "feature_contract": FEATURE_CONTRACT_VERSION,
        "features": list(FEATURE_COLUMNS),
        "n_train": int(n_train),
        "n_valid": int(n_valid),
        "pr_auc_train": pr_auc_train,
        "pr_auc_valid": pr_auc_valid,
        "overfit_gap": overfit_gap,
        "best_threshold": best_threshold,
        "best_precision": best_metrics["precision"],
        "best_recall": best_metrics["recall"],
        "best_fbeta": best_metrics["fbeta"],
        "beta": train_config.beta,
        "dropped_train_rows_null_outputs": int(train_rows_before_filter - n_train),
        "dropped_valid_rows_null_outputs": int(valid_rows_before_filter - n_valid),
        "model_artifact_path": "model",
    }


def run_training_pipeline(data_config: DataConfig, train_config: TrainConfig) -> dict:
    pg_config = build_pg_config(data_config)
    engine = create_pg_engine(pg_config)
    dt_list = list_unique_dt_postgres(
        engine,
        pg_config,
        city=data_config.city,
        start_dt=normalize_timestamp(data_config.start),
        end_dt=normalize_timestamp(data_config.end),
    )
    split = compute_temporal_split(dt_list, train_config.valid_ratio, train_config.gap_minutes)
    select_columns = dedupe_columns([*REQUIRED_BASE, *FEATURE_COLUMNS, *LABEL_COLUMNS, "city", "dt", "station_id"])

    train_df = load_slice_postgres(
        engine,
        data_config,
        select_columns,
        normalize_timestamp(data_config.start),
        split.train_end_dt,
    )
    valid_df = load_slice_postgres(
        engine,
        data_config,
        select_columns,
        split.valid_start_dt,
        normalize_timestamp(data_config.end),
    )

    target_spec = target_spec_from_predict_bikes(train_config.predict_bikes)
    train_rows_before_filter = len(train_df)
    valid_rows_before_filter = len(valid_df)
    train_df = train_df.dropna(subset=[target_spec.label_column, target_spec.paired_target_column]).copy()
    valid_df = valid_df.dropna(subset=[target_spec.label_column, target_spec.paired_target_column]).copy()
    if train_df.empty or valid_df.empty:
        raise RuntimeError("train or validation slice is empty after dropping null labels")

    x_train = train_df[FEATURE_COLUMNS].astype("float64")
    y_train = train_df[target_spec.label_column].astype(int).to_numpy()
    x_valid = valid_df[FEATURE_COLUMNS].astype("float64")
    y_valid = valid_df[target_spec.label_column].astype(int).to_numpy()
    _ensure_binary_classes(y_train, "train", target_spec.label_column)
    _ensure_binary_classes(y_valid, "validation", target_spec.label_column)

    model_name = build_model_name(data_config.city, target_spec.label_column, train_config.model_type)
    feature_source = build_feature_source_name(data_config)
    plot_dir = f"plots/{model_name}"
    eval_path = "eval/eval_summary.json"
    importance_dir = f"feature_importance/{model_name}"

    mlflow.set_experiment(train_config.experiment)
    mlflow.autolog(disable=True)

    with mlflow.start_run(run_name=f"train-{model_name}") as run:
        model = build_classifier(train_config)
        mlflow.set_tags(
            {
                "city": data_config.city,
                "label": target_spec.label_column,
                "target_name": target_spec.target_name,
                "predict_bikes": str(target_spec.predict_bikes).lower(),
                "model_type": train_config.model_type,
                "model_name": model_name,
                "feature_source": feature_source,
                "feature_contract": FEATURE_CONTRACT_VERSION,
                "run_reason": train_config.run_reason,
                "train_window_start": data_config.start,
                "train_window_end": data_config.end,
            }
        )
        mlflow.log_params(
            {
                "valid_ratio": train_config.valid_ratio,
                "gap_minutes": train_config.gap_minutes,
                "beta": train_config.beta,
                "pg_schema": data_config.pg_schema,
                "feature_table": data_config.feature_table,
            }
        )

        model.fit(x_train, y_train)
        train_prob = model.predict_proba(x_train)[:, 1]
        valid_prob = model.predict_proba(x_valid)[:, 1]

        pr_auc_train = float(average_precision_score(y_train, train_prob))
        pr_auc_valid = float(average_precision_score(y_valid, valid_prob))
        overfit_gap = abs(pr_auc_train - pr_auc_valid)
        best_threshold, best_metrics = pick_threshold_fbeta(y_valid, valid_prob, beta=train_config.beta)

        mlflow.log_metrics(
            {
                "pr_auc_train": pr_auc_train,
                "pr_auc_valid": pr_auc_valid,
                "overfit_gap": overfit_gap,
                "best_precision": best_metrics["precision"],
                "best_recall": best_metrics["recall"],
                "best_fbeta": best_metrics["fbeta"],
                "best_threshold": best_threshold,
                "n_train": len(x_train),
                "n_valid": len(x_valid),
                "dropped_train_rows_null_outputs": train_rows_before_filter - len(train_df),
                "dropped_valid_rows_null_outputs": valid_rows_before_filter - len(valid_df),
            }
        )

        log_curves_and_confusion(y_train, train_prob, best_threshold, plot_dir, prefix="train")
        log_curves_and_confusion(y_valid, valid_prob, best_threshold, plot_dir, prefix="valid")
        log_feature_importance(model, FEATURE_COLUMNS, importance_dir)

        sample_x = build_sample_input_frame(FEATURE_COLUMNS)
        signature = infer_signature(sample_x, pd.Series([0.5], dtype="float64"))
        _save_or_log_pyfunc_probability_model(
            model=model,
            model_type=train_config.model_type,
            feature_names=FEATURE_COLUMNS,
            signature=signature,
            input_example=sample_x,
            log_name="model",
        )

        eval_summary = build_eval_summary(
            run_id=run.info.run_id,
            train_config=train_config,
            data_config=data_config,
            target_spec=target_spec,
            feature_source=feature_source,
            split=split,
            n_train=len(x_train),
            n_valid=len(x_valid),
            train_rows_before_filter=train_rows_before_filter,
            valid_rows_before_filter=valid_rows_before_filter,
            pr_auc_train=pr_auc_train,
            pr_auc_valid=pr_auc_valid,
            overfit_gap=overfit_gap,
            best_threshold=best_threshold,
            best_metrics=best_metrics,
        )
        mlflow.log_dict(eval_summary, eval_path)

        package_dir, package_manifest_path = _save_local_package(
            eval_summary=eval_summary,
            run=run,
            model=model,
            package_root=train_config.package_root,
        )
        eval_summary["package_dir"] = package_dir
        eval_summary["package_manifest_path"] = package_manifest_path
        mlflow.log_text(Path(package_manifest_path).read_text(encoding="utf-8"), "package_manifest.json")
        mlflow.log_text(build_model_card_text(eval_summary), "artifacts/model_card.md")

    return eval_summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Postgres-backed offline stockout model.")
    parser.add_argument("--city", required=True)
    parser.add_argument("--start", required=True, help="UTC window start in 'YYYY-MM-DD HH:MM'")
    parser.add_argument("--end", required=True, help="UTC window end in 'YYYY-MM-DD HH:MM'")
    parser.add_argument("--pg-host", default=env_or_default("PGHOST"))
    parser.add_argument("--pg-port", default=env_or_default("PGPORT", "5432"), type=int)
    parser.add_argument("--pg-db", default=env_or_default("PGDATABASE"))
    parser.add_argument("--pg-user", default=env_or_default("PGUSER"))
    parser.add_argument("--pg-password", default=env_or_default("PGPASSWORD"))
    parser.add_argument("--pg-schema", default=env_or_default("PGSCHEMA", "analytics"))
    parser.add_argument("--feature-table", default=env_or_default("FEATURE_TABLE", "feat_station_snapshot_5min"))
    parser.add_argument("--predict-bikes", default=env_or_default("PREDICT_BIKES", "true"), choices=["true", "false"])
    parser.add_argument("--model-type", default="xgboost", choices=["xgboost", "lightgbm"])
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--gap-minutes", type=int, default=60)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--experiment",
        default=env_or_default("TRAINING_EXPERIMENT", "bikeshare-step4"),
    )
    parser.add_argument(
        "--package-root", default=None, help="Optional package root. Defaults to model_dir/packages/<target>."
    )
    parser.add_argument(
        "--run-reason",
        default="manual",
        choices=["schedule", "manual", "drift", "post_rollback"],
    )
    args = parser.parse_args(argv)
    missing = [name for name in ("pg_host", "pg_db", "pg_user", "pg_password") if getattr(args, name) in {None, ""}]
    if missing:
        raise ValueError(f"missing required Postgres settings: {missing}")
    return args


def env_or_default(key: str, default: str | None = None) -> str | None:
    value = os.environ.get(key)
    if value not in {None, ""}:
        return value
    return default


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    predict_bikes = parse_bool_value(args.predict_bikes, default=True)
    data_config = DataConfig(
        city=args.city,
        start=args.start,
        end=args.end,
        pg_host=args.pg_host,
        pg_port=args.pg_port,
        pg_db=args.pg_db,
        pg_user=args.pg_user,
        pg_password=args.pg_password,
        pg_schema=args.pg_schema,
        feature_table=args.feature_table,
    )
    train_config = TrainConfig(
        predict_bikes=predict_bikes,
        model_type=args.model_type,
        valid_ratio=args.valid_ratio,
        gap_minutes=args.gap_minutes,
        random_state=args.random_state,
        beta=args.beta,
        experiment=args.experiment,
        run_reason=args.run_reason,
        package_root=args.package_root
        or str(default_package_root_for_target(target_spec_from_predict_bikes(predict_bikes).target_name)),
    )
    result = run_training_pipeline(data_config, train_config)
    print(TRAINING_RESULT_PREFIX + json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
