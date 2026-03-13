# scripts/publish_custom_metrics.py
# Purpose: Load the last 24h of model-quality data from S3 (parquet),
# compute PR-AUC, F1 at a fixed threshold, threshold hit-rate, and publish
# them as CloudWatch custom metrics under the "Bikeshare/Model" namespace.

import argparse
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import boto3
import numpy as np
import pandas as pd

from src.config import quality_prefix, resolve_target_name
from src.model_target import parse_bool_value, target_spec_from_name, target_spec_from_predict_bikes
from src.monitoring.metrics.metrics_helper import put_metrics_bulk

# ---------- Metric helpers (no heavy sklearn dependency) ----------


def _auto_pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column that exists in df from candidates, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def compute_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Average Precision (area under precision-recall curve) in pure numpy.

    AP = sum_k( (R_k - R_{k-1}) * P_k ), where k runs over indices of positives
    after sorting by score descending.
    """
    if y_true.size == 0:
        return 0.0
    y_true = y_true.astype(np.int8)
    y_score = y_score.astype(np.float64)

    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]

    tp_cum = np.cumsum(y_true_sorted)
    ranks = np.arange(1, y_true_sorted.size + 1, dtype=np.float64)

    precision = tp_cum / ranks
    total_pos = max(1, int(y_true.sum()))
    recall = tp_cum / float(total_pos)

    pos_idx = np.where(y_true_sorted == 1)[0]
    if pos_idx.size == 0:
        return 0.0

    recall_at_pos = recall[pos_idx]
    precision_at_pos = precision[pos_idx]
    recall_prev = np.concatenate(([0.0], recall_at_pos[:-1]))
    delta = recall_at_pos - recall_prev
    ap = np.sum(precision_at_pos * delta)
    return float(ap)


def compute_f1_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> float:
    """Compute F1 score at a fixed probability threshold."""
    y_pred = (y_score >= thr).astype(np.int8)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2.0 * tp) / denom


def compute_threshold_hit_rate(y_score: np.ndarray, thr: float) -> float:
    """Fraction of predictions >= thr."""
    if y_score.size == 0:
        return 0.0
    return float(np.mean(y_score >= thr))


# ---------- S3 + Data loading ----------


def list_parquet_keys_for_last_24h(s3, bucket: str, prefix_root: str) -> List[str]:
    """
    List object keys under prefix_root limited to partitions for the last 24h.
    Example prefix_root: monitoring/quality/city=paris
    We include 'ds=<today>' and 'ds=<yesterday>'.
    """
    now = datetime.now(timezone.utc)
    days = {now.date(), (now - timedelta(days=1)).date()}
    prefixes = [f"{prefix_root.rstrip('/')}/ds={d.isoformat()}/" for d in sorted(days)]

    keys = []
    for p in prefixes:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=p):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".parquet"):
                    keys.append(obj["Key"])
    return keys


def load_quality_dataframe(bucket: str, prefix_root: str, time_col_hint: str = "dt") -> pd.DataFrame:
    """
    Load and concatenate parquet files from the last 24h.
    We filter by time_col_hint (default 'dt') if present.
    """
    s3 = boto3.client("s3")
    keys = list_parquet_keys_for_last_24h(s3, bucket, prefix_root)
    if not keys:
        return pd.DataFrame()

    paths = [f"s3://{bucket}/{k}" for k in keys]
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)

    # Prefer the provided time column (dt). If missing, try common alternatives.
    time_col = time_col_hint if time_col_hint in df.columns else None
    if time_col is None:
        for c in ["timestamp", "event_time", "prediction_time", "inference_time"]:
            if c in df.columns:
                time_col = c
                break

    if time_col is not None:
        # dt like "2025-09-25 01:55:00" (no timezone)
        df[time_col] = pd.to_datetime(
            df[time_col], format="%Y-%m-%d-%H-%M", utc=True, errors="coerce"  # exact format for "YYYY-MM-DD-HH-MM"
        )
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        df = df[df[time_col] >= cutoff].copy()

    return df


def resolve_quality_columns(df: pd.DataFrame, predict_bikes: bool | None = None) -> tuple[str, str]:
    if predict_bikes is not None:
        target_spec = target_spec_from_predict_bikes(predict_bikes)
        return target_spec.label_column, target_spec.score_column

    if "prediction_target" in df.columns:
        target_values = df["prediction_target"].dropna().astype(str).str.lower().unique().tolist()
        if len(target_values) == 1:
            target_spec = target_spec_from_name(target_values[0])
            return target_spec.label_column, target_spec.score_column
        if len(target_values) > 1:
            raise ValueError(f"quality data must contain exactly one prediction_target, got {target_values}")

    if {"y_stockout_bikes_30", "y_stockout_docks_30"}.issubset(df.columns):
        raise RuntimeError("quality data exposes both label columns without prediction_target disambiguation")
    if {"yhat_bikes", "yhat_docks"}.issubset(df.columns):
        raise RuntimeError("quality data exposes both score columns without prediction_target disambiguation")

    label_col = _auto_pick_column(df, ["y_stockout_bikes_30", "y_stockout_docks_30"])
    score_col = _auto_pick_column(df, ["yhat_bikes", "yhat_docks"])
    if label_col is None or score_col is None:
        raise RuntimeError(f"Expected side-specific label/score columns not found. Have: {list(df.columns)}")
    return label_col, score_col


# ---------- Main ----------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="S3 bucket containing quality data")
    parser.add_argument(
        "--quality-prefix",
        required=True,
        help="S3 prefix root that contains ds=YYYY-MM-DD partitions, e.g., monitoring/quality/target=bikes/city=paris",
    )
    parser.add_argument("--endpoint", required=True, help="Endpoint name as a metric dimension")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "eu-west-3"))
    parser.add_argument("--threshold", type=float, default=0.15, help="Probability threshold for F1/hit-rate")
    parser.add_argument("--namespace", default="Bikeshare/Model")
    parser.add_argument("--city-dimension", default="paris", help="Optional extra CloudWatch dimension City=<value>")
    parser.add_argument("--predict-bikes", default=None, choices=["true", "false"])
    parser.add_argument("--target-name", default=None, choices=["bikes", "docks"])
    parser.add_argument("--environment", default=os.environ.get("SERVING_ENVIRONMENT", "production"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute metrics and print the payload without publishing to CloudWatch.",
    )
    args = parser.parse_args()

    target_name = resolve_target_name(
        predict_bikes=args.predict_bikes,
        target_name=args.target_name,
    )
    quality_root = args.quality_prefix
    if quality_root == "AUTO":
        quality_root = quality_prefix(args.city_dimension, target_name)

    # Load data (use 'dt' as the time column)
    df = load_quality_dataframe(args.bucket, quality_root, time_col_hint="dt")
    if df.empty:
        print("No quality data found for the last 24h. Skipping metric publish.")
        return

    predict_bikes = None if args.predict_bikes is None else parse_bool_value(args.predict_bikes)
    label_col, score_col = resolve_quality_columns(df, predict_bikes)

    y_true = df[label_col].astype(int).to_numpy(copy=False)
    y_score = df[score_col].astype(float).to_numpy(copy=False)

    # Compute metrics
    pr_auc = compute_average_precision(y_true, y_score)
    f1 = compute_f1_at_threshold(y_true, y_score, args.threshold)
    hit_rate = compute_threshold_hit_rate(y_score, args.threshold)
    samples = int(y_true.size)

    print(
        f"Computed metrics (last 24h): AP={pr_auc:.4f}, F1@{args.threshold}={f1:.4f}, "
        f"HitRate@{args.threshold}={hit_rate:.4f}, N={samples}"
    )

    metric_payload = [
        ("PR-AUC-24h", pr_auc, "None"),
        ("F1-24h", f1, "None"),
        ("ThresholdHitRate-24h", hit_rate, "None"),
        ("Samples-24h", samples, "Count"),
    ]
    if args.dry_run:
        print(
            "Dry run only. Prepared 4 metrics for namespace "
            f"'{args.namespace}' with dimensions "
            f"Environment={args.environment}, EndpointName={args.endpoint}, City={args.city_dimension}, TargetName={target_name}."
        )
        return

    put_metrics_bulk(
        metric_payload,
        endpoint=args.endpoint,
        city=args.city_dimension,
        target_name=target_name,
        environment=args.environment,
    )
    print(
        "Published 4 metrics to CloudWatch namespace "
        f"'{args.namespace}' for target={target_name}, environment={args.environment}, endpoint={args.endpoint}."
    )


if __name__ == "__main__":
    main()
