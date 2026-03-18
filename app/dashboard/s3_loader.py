"""S3 Parquet loader for dashboard predictions and quality data.

Replaces the former Athena-based queries. Reads directly from S3 using
boto3 + pandas, following the same key structure written by predictor.py
and quality_backfill.py.
"""
from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import botocore.exceptions

import pandas as pd

from src.config.naming import prediction_prefix, quality_prefix
from src.model_target import target_spec_from_name
from src.config import resolve_target_name

if TYPE_CHECKING:
    import boto3


def _list_dt_keys(s3_client, bucket: str, prefix: str) -> list[str]:
    """Return all object keys under a given S3 prefix, sorted ascending.

    Returns an empty list if the bucket does not exist or is inaccessible.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    keys = []
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
    except botocore.exceptions.ClientError:
        return []
    return sorted(keys)


def _read_parquet_from_s3(s3_client, bucket: str, key: str) -> pd.DataFrame:
    """Download a single Parquet file from S3 and return as DataFrame."""
    resp = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(resp["Body"].read()))


def _parse_dt(dt_str: str) -> datetime | None:
    """Parse dt string in YYYY-MM-DD-HH-MM format to UTC datetime."""
    try:
        return datetime.strptime(dt_str, "%Y-%m-%d-%H-%M").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def load_latest_predictions(
    *,
    bucket: str,
    city: str,
    target_name: str,
    s3_client,
) -> pd.DataFrame:
    """Load the most recent prediction Parquet file for a given target.

    Returns DataFrame with columns: station_id (str), ts (datetime), score (float).
    Returns empty DataFrame if no predictions found.
    """
    target = resolve_target_name(target_name=target_name)
    spec = target_spec_from_name(target)
    prefix = prediction_prefix(city, target)

    keys = _list_dt_keys(s3_client, bucket, prefix)
    if not keys:
        return pd.DataFrame(columns=["station_id", "ts", "score"])

    latest_key = keys[-1]
    try:
        df = _read_parquet_from_s3(s3_client, bucket, latest_key)
    except Exception:
        return pd.DataFrame(columns=["station_id", "ts", "score"])

    if df.empty or spec.score_column not in df.columns:
        return pd.DataFrame(columns=["station_id", "ts", "score"])

    result = pd.DataFrame()
    result["station_id"] = df["station_id"].astype(str)
    result["ts"] = df["dt"].apply(_parse_dt)
    result["score"] = pd.to_numeric(df[spec.score_column], errors="coerce").clip(0.0, 1.0)
    return result.dropna(subset=["score"])


def load_prediction_history(
    *,
    bucket: str,
    city: str,
    target_name: str,
    station_id: str,
    n_periods: int,
    s3_client,
) -> pd.DataFrame:
    """Load prediction history for a single station from the last n_periods snapshots.

    Returns DataFrame with columns: station_id (str), ts (datetime), score (float).
    """
    target = resolve_target_name(target_name=target_name)
    spec = target_spec_from_name(target)
    prefix = prediction_prefix(city, target)

    keys = _list_dt_keys(s3_client, bucket, prefix)
    if not keys:
        return pd.DataFrame(columns=["station_id", "ts", "score"])

    recent_keys = keys[-n_periods:]
    frames = []
    for key in recent_keys:
        try:
            df = _read_parquet_from_s3(s3_client, bucket, key)
            if df.empty or spec.score_column not in df.columns:
                continue
            station_rows = df[df["station_id"].astype(str) == str(station_id)]
            if station_rows.empty:
                continue
            chunk = pd.DataFrame()
            chunk["station_id"] = station_rows["station_id"].astype(str)
            chunk["ts"] = station_rows["dt"].apply(_parse_dt)
            chunk["score"] = pd.to_numeric(station_rows[spec.score_column], errors="coerce").clip(0.0, 1.0)
            frames.append(chunk)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=["station_id", "ts", "score"])
    return pd.concat(frames, ignore_index=True).dropna(subset=["score"])


def load_quality_recent(
    *,
    bucket: str,
    city: str,
    target_name: str,
    s3_client,
    lookback_hours: int = 24,
) -> pd.DataFrame:
    """Load recent quality metric rows (last lookback_hours) for model health display.

    Returns DataFrame with columns: dt, score (float), label (int).
    """
    target = resolve_target_name(target_name=target_name)
    spec = target_spec_from_name(target)
    prefix = quality_prefix(city, target)

    # List quality files covering today and (if lookback spans midnight) yesterday
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=lookback_hours)
    keys = _list_dt_keys(s3_client, bucket, prefix)
    if not keys:
        return pd.DataFrame(columns=["dt", "score", "label"])

    frames = []
    for key in keys:
        try:
            df = _read_parquet_from_s3(s3_client, bucket, key)
            if df.empty:
                continue
            if spec.score_column not in df.columns or spec.label_column not in df.columns:
                continue
            chunk = pd.DataFrame()
            chunk["dt"] = df["dt"]
            chunk["score"] = pd.to_numeric(df[spec.score_column], errors="coerce")
            chunk["label"] = pd.to_numeric(df[spec.label_column], errors="coerce").astype("Int64")
            frames.append(chunk)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=["dt", "score", "label"])

    combined = pd.concat(frames, ignore_index=True)
    combined["ts"] = combined["dt"].apply(_parse_dt)
    combined = combined.dropna(subset=["ts"])
    return combined[combined["ts"] >= cutoff].reset_index(drop=True)
