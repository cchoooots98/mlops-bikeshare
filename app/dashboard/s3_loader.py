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

from dashboard.contracts import ArtifactLoadResult, LoadStatus
from src.config.naming import prediction_prefix, quality_prefix
from src.config import resolve_target_name
from src.model_target import target_spec_from_name

if TYPE_CHECKING:
    import boto3


def _status_from_client_error(exc: botocore.exceptions.ClientError) -> tuple[LoadStatus, str]:
    code = exc.response.get("Error", {}).get("Code", "")
    message = exc.response.get("Error", {}).get("Message", str(exc))
    if code in {"AccessDenied", "AccessDeniedException"}:
        return LoadStatus.ACCESS_DENIED, f"AWS access denied: {message}"
    return LoadStatus.READ_ERROR, f"AWS read error ({code or 'unknown'}): {message}"


def _status_from_exception(exc: Exception) -> tuple[LoadStatus, str]:
    if isinstance(exc, botocore.exceptions.ClientError):
        return _status_from_client_error(exc)
    return LoadStatus.READ_ERROR, f"Unable to read S3 artifact: {exc}"


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


def _list_keys_detailed(s3_client, bucket: str, prefix: str) -> tuple[list[str], LoadStatus, str]:
    paginator = s3_client.get_paginator("list_objects_v2")
    keys: list[str] = []
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
    except botocore.exceptions.ClientError as exc:
        status, message = _status_from_client_error(exc)
        return [], status, message
    return sorted(keys), LoadStatus.OK, ""


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


def _latest_dt_from_df(df: pd.DataFrame) -> datetime | None:
    if "dt" not in df.columns or df.empty:
        return None
    parsed = df["dt"].apply(_parse_dt).dropna()
    if parsed.empty:
        return None
    return max(parsed)


def _build_result(
    *,
    status: LoadStatus,
    source_name: str,
    data: pd.DataFrame | None = None,
    message: str = "",
    latest_key: str | None = None,
    row_count: int = 0,
    valid_score_count: int = 0,
    latest_dt: datetime | None = None,
) -> ArtifactLoadResult:
    return ArtifactLoadResult(
        status=status,
        data=data if data is not None else pd.DataFrame(),
        message=message,
        latest_key=latest_key,
        row_count=row_count,
        valid_score_count=valid_score_count,
        latest_dt=latest_dt,
        source_name=source_name,
    )


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

    keys, status, message = _list_keys_detailed(s3_client, bucket, prefix)
    if status != LoadStatus.OK:
        return _build_result(status=status, source_name="Prediction artifact", message=message)
    if not keys:
        return _build_result(
            status=LoadStatus.NO_OBJECTS,
            source_name="Prediction artifact",
            message=f"No prediction artifacts found under {prefix}",
        )

    latest_key = keys[-1]
    try:
        df = _read_parquet_from_s3(s3_client, bucket, latest_key)
    except Exception as exc:
        result_status, result_message = _status_from_exception(exc)
        return _build_result(
            status=result_status,
            source_name="Prediction artifact",
            message=result_message,
            latest_key=latest_key,
        )

    if df.empty or spec.score_column not in df.columns:
        return _build_result(
            status=LoadStatus.SCHEMA_ERROR,
            source_name="Prediction artifact",
            message=f"Prediction artifact is missing the required score column '{spec.score_column}'.",
            latest_key=latest_key,
            row_count=len(df),
            latest_dt=_latest_dt_from_df(df),
        )

    result = pd.DataFrame()
    result["station_id"] = df["station_id"].astype(str)
    result["ts"] = df["dt"].apply(_parse_dt)
    result["score"] = pd.to_numeric(df[spec.score_column], errors="coerce").clip(0.0, 1.0)
    filtered = result.dropna(subset=["score"]).reset_index(drop=True)
    valid_score_count = int(filtered["score"].notna().sum())
    latest_dt = _latest_dt_from_df(df)
    if valid_score_count == 0:
        return _build_result(
            status=LoadStatus.ALL_SCORES_NULL,
            source_name="Prediction artifact",
            message="Prediction artifact exists, but every prediction score is null or invalid. Upstream inference likely failed.",
            latest_key=latest_key,
            row_count=len(df),
            valid_score_count=0,
            latest_dt=latest_dt,
        )

    return _build_result(
        status=LoadStatus.OK,
        source_name="Prediction artifact",
        data=filtered,
        latest_key=latest_key,
        row_count=len(df),
        valid_score_count=valid_score_count,
        latest_dt=latest_dt,
    )


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

    keys, status, message = _list_keys_detailed(s3_client, bucket, prefix)
    if status != LoadStatus.OK:
        return _build_result(status=status, source_name="Prediction history", message=message)
    if not keys:
        return _build_result(
            status=LoadStatus.NO_OBJECTS,
            source_name="Prediction history",
            message=f"No prediction history found under {prefix}",
        )

    recent_keys = keys[-n_periods:]
    frames = []
    rows_seen = 0
    latest_dt = None
    for key in recent_keys:
        try:
            df = _read_parquet_from_s3(s3_client, bucket, key)
        except Exception as exc:
            result_status, result_message = _status_from_exception(exc)
            return _build_result(
                status=result_status,
                source_name="Prediction history",
                message=result_message,
                latest_key=key,
            )
        if df.empty or spec.score_column not in df.columns:
            return _build_result(
                status=LoadStatus.SCHEMA_ERROR,
                source_name="Prediction history",
                message=f"History artifact is missing the required score column '{spec.score_column}'.",
                latest_key=key,
                row_count=len(df),
                latest_dt=_latest_dt_from_df(df),
            )
        rows_seen += len(df)
        latest_dt = _latest_dt_from_df(df) or latest_dt
        station_rows = df[df["station_id"].astype(str) == str(station_id)]
        if station_rows.empty:
            continue
        chunk = pd.DataFrame()
        chunk["station_id"] = station_rows["station_id"].astype(str)
        chunk["ts"] = station_rows["dt"].apply(_parse_dt)
        chunk["score"] = pd.to_numeric(station_rows[spec.score_column], errors="coerce").clip(0.0, 1.0)
        frames.append(chunk)

    if not frames:
        return _build_result(
            status=LoadStatus.NO_OBJECTS,
            source_name="Prediction history",
            message=f"No history snapshots found yet for station {station_id}.",
            latest_key=recent_keys[-1],
            row_count=rows_seen,
            latest_dt=latest_dt,
        )

    combined = pd.concat(frames, ignore_index=True).dropna(subset=["score"]).reset_index(drop=True)
    valid_score_count = int(combined["score"].notna().sum())
    if valid_score_count == 0:
        return _build_result(
            status=LoadStatus.ALL_SCORES_NULL,
            source_name="Prediction history",
            message=f"History artifacts exist for station {station_id}, but all prediction scores are invalid.",
            latest_key=recent_keys[-1],
            row_count=rows_seen,
            latest_dt=latest_dt,
        )

    return _build_result(
        status=LoadStatus.OK,
        source_name="Prediction history",
        data=combined,
        latest_key=recent_keys[-1],
        row_count=rows_seen,
        valid_score_count=valid_score_count,
        latest_dt=latest_dt,
    )


def load_latest_quality_status(
    *,
    bucket: str,
    city: str,
    target_name: str,
    s3_client,
) -> ArtifactLoadResult:
    target = resolve_target_name(target_name=target_name)
    spec = target_spec_from_name(target)
    prefix = quality_prefix(city, target)

    keys, status, message = _list_keys_detailed(s3_client, bucket, prefix)
    if status != LoadStatus.OK:
        return _build_result(status=status, source_name="Quality artifact", message=message)
    if not keys:
        return _build_result(
            status=LoadStatus.NO_OBJECTS,
            source_name="Quality artifact",
            message=f"No quality artifacts found under {prefix}",
        )

    latest_key = keys[-1]
    try:
        df = _read_parquet_from_s3(s3_client, bucket, latest_key)
    except Exception as exc:
        result_status, result_message = _status_from_exception(exc)
        return _build_result(
            status=result_status,
            source_name="Quality artifact",
            message=result_message,
            latest_key=latest_key,
        )

    if df.empty or spec.score_column not in df.columns or spec.label_column not in df.columns:
        return _build_result(
            status=LoadStatus.SCHEMA_ERROR,
            source_name="Quality artifact",
            message=(
                "Quality artifact is missing required target-aware score/label columns "
                f"('{spec.score_column}', '{spec.label_column}')."
            ),
            latest_key=latest_key,
            row_count=len(df),
            latest_dt=_latest_dt_from_df(df),
        )

    valid_score_count = int(pd.to_numeric(df[spec.score_column], errors="coerce").notna().sum())
    latest_dt = _latest_dt_from_df(df)
    if valid_score_count == 0:
        return _build_result(
            status=LoadStatus.ALL_SCORES_NULL,
            source_name="Quality artifact",
            message="Quality artifact exists, but all scores are null or invalid.",
            latest_key=latest_key,
            row_count=len(df),
            latest_dt=latest_dt,
        )

    return _build_result(
        status=LoadStatus.OK,
        source_name="Quality artifact",
        data=df,
        latest_key=latest_key,
        row_count=len(df),
        valid_score_count=valid_score_count,
        latest_dt=latest_dt,
    )


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
