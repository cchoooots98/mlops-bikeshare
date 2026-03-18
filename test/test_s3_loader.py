"""Unit tests for app/dashboard/s3_loader.py.

Uses unittest.mock to simulate boto3 S3 responses.
No AWS credentials or network access required.
"""
import io
import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest
from botocore.exceptions import ClientError

# Mirror Streamlit's runtime: add app/ to sys.path so `dashboard.*` is importable.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP_DIR = os.path.join(_REPO_ROOT, "app")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from dashboard.s3_loader import (  # noqa: E402
    _list_dt_keys,
    _parse_dt,
    load_latest_predictions,
    load_prediction_history,
    load_quality_recent,
)

# ── Fixtures / helpers ────────────────────────────────────────────────


def _make_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _prediction_df(
    station_ids: list[str],
    dt: str,
    target: str = "bikes",
) -> pd.DataFrame:
    score_col = "yhat_bikes" if target == "bikes" else "yhat_docks"
    return pd.DataFrame({
        "station_id": station_ids,
        "dt": [dt] * len(station_ids),
        score_col: [0.75] * len(station_ids),
    })


def _quality_df(
    station_ids: list[str],
    dt: str,
    target: str = "bikes",
) -> pd.DataFrame:
    score_col = "yhat_bikes" if target == "bikes" else "yhat_docks"
    label_col = "y_stockout_bikes_30" if target == "bikes" else "y_stockout_docks_30"
    return pd.DataFrame({
        "station_id": station_ids,
        "dt": [dt] * len(station_ids),
        score_col: [0.6] * len(station_ids),
        label_col: [0] * len(station_ids),
    })


def _s3_mock(keys: list[str], parquet_bytes: bytes | None = None) -> MagicMock:
    """S3 mock that returns the given keys from the paginator."""
    client = MagicMock()
    paginator = MagicMock()
    pages = [{"Contents": [{"Key": k} for k in keys]}] if keys else [{}]
    paginator.paginate.return_value = pages
    client.get_paginator.return_value = paginator
    if parquet_bytes is not None:
        client.get_object.return_value = {"Body": io.BytesIO(parquet_bytes)}
    return client


def _s3_mock_error(code: str = "NoSuchBucket") -> MagicMock:
    """S3 mock that raises ClientError on paginate."""
    client = MagicMock()
    paginator = MagicMock()
    paginator.paginate.side_effect = ClientError(
        {"Error": {"Code": code, "Message": "mocked error"}},
        "ListObjectsV2",
    )
    client.get_paginator.return_value = paginator
    return client


# ── _parse_dt ─────────────────────────────────────────────────────────


def test_parse_dt_valid():
    result = _parse_dt("2026-03-18-15-30")
    assert result == datetime(2026, 3, 18, 15, 30, tzinfo=timezone.utc)


def test_parse_dt_invalid_returns_none():
    assert _parse_dt("not-a-date") is None
    assert _parse_dt("") is None
    assert _parse_dt(None) is None


# ── _list_dt_keys ─────────────────────────────────────────────────────


def test_list_dt_keys_returns_sorted_keys():
    keys = [
        "inference/target=bikes/city=paris/dt=2026-03-18-15-30/predictions.parquet",
        "inference/target=bikes/city=paris/dt=2026-03-18-15-25/predictions.parquet",
    ]
    client = _s3_mock(keys)
    result = _list_dt_keys(client, "my-bucket", "inference/target=bikes/city=paris/dt=")
    assert result == sorted(keys)


def test_list_dt_keys_returns_empty_on_no_such_bucket():
    client = _s3_mock_error("NoSuchBucket")
    result = _list_dt_keys(client, "nonexistent-bucket", "some/prefix/")
    assert result == []


def test_list_dt_keys_returns_empty_on_access_denied():
    client = _s3_mock_error("AccessDenied")
    result = _list_dt_keys(client, "my-bucket", "some/prefix/")
    assert result == []


def test_list_dt_keys_returns_empty_when_no_objects():
    client = _s3_mock([])
    result = _list_dt_keys(client, "my-bucket", "empty/prefix/")
    assert result == []


# ── load_latest_predictions ───────────────────────────────────────────


def test_load_latest_predictions_returns_correct_schema():
    dt = "2026-03-18-15-30"
    df = _prediction_df(["s1", "s2"], dt)
    parquet = _make_parquet_bytes(df)
    key = f"inference/target=bikes/city=paris/dt={dt}/predictions.parquet"
    client = _s3_mock([key], parquet)

    result = load_latest_predictions(bucket="b", city="paris", target_name="bikes", s3_client=client)

    assert list(result.columns) == ["station_id", "ts", "score"]
    assert result["station_id"].dtype == object  # str
    assert len(result) == 2
    assert result["score"].between(0.0, 1.0).all()


def test_load_latest_predictions_picks_last_key():
    keys = [
        "inference/target=bikes/city=paris/dt=2026-03-18-15-25/predictions.parquet",
        "inference/target=bikes/city=paris/dt=2026-03-18-15-30/predictions.parquet",
    ]
    dt_latest = "2026-03-18-15-30"
    df = _prediction_df(["s1"], dt_latest)
    parquet = _make_parquet_bytes(df)
    client = _s3_mock(keys, parquet)

    result = load_latest_predictions(bucket="b", city="paris", target_name="bikes", s3_client=client)

    # Verify get_object was called with the last (most recent) key
    call_args = client.get_object.call_args
    assert call_args.kwargs["Key"] == keys[-1]
    assert len(result) == 1


def test_load_latest_predictions_bucket_not_found_returns_empty():
    client = _s3_mock_error("NoSuchBucket")
    result = load_latest_predictions(bucket="bad-bucket", city="paris", target_name="bikes", s3_client=client)
    assert result.empty
    assert list(result.columns) == ["station_id", "ts", "score"]


def test_load_latest_predictions_empty_bucket_returns_empty():
    client = _s3_mock([])
    result = load_latest_predictions(bucket="b", city="paris", target_name="bikes", s3_client=client)
    assert result.empty


def test_load_latest_predictions_missing_score_column_returns_empty():
    df = pd.DataFrame({"station_id": ["s1"], "dt": ["2026-03-18-15-30"], "other_col": [1.0]})
    parquet = _make_parquet_bytes(df)
    key = "inference/target=bikes/city=paris/dt=2026-03-18-15-30/predictions.parquet"
    client = _s3_mock([key], parquet)

    result = load_latest_predictions(bucket="b", city="paris", target_name="bikes", s3_client=client)
    assert result.empty


# ── load_prediction_history ───────────────────────────────────────────


def test_load_prediction_history_filters_by_station_id():
    dt = "2026-03-18-15-30"
    df = _prediction_df(["s1", "s2", "s3"], dt)
    parquet = _make_parquet_bytes(df)
    key = f"inference/target=bikes/city=paris/dt={dt}/predictions.parquet"
    client = _s3_mock([key], parquet)

    result = load_prediction_history(
        bucket="b", city="paris", target_name="bikes", station_id="s2", n_periods=5, s3_client=client
    )

    assert len(result) == 1
    assert result.iloc[0]["station_id"] == "s2"


def test_load_prediction_history_limits_to_n_periods():
    keys = [
        f"inference/target=bikes/city=paris/dt=2026-03-18-15-{i:02d}/predictions.parquet"
        for i in range(10)
    ]
    dt = "2026-03-18-15-05"
    df = _prediction_df(["s1"], dt)
    parquet = _make_parquet_bytes(df)
    client = _s3_mock(keys, parquet)

    load_prediction_history(
        bucket="b", city="paris", target_name="bikes", station_id="s1", n_periods=3, s3_client=client
    )

    # get_object should be called at most 3 times (last 3 keys)
    assert client.get_object.call_count <= 3


def test_load_prediction_history_bucket_not_found_returns_empty():
    client = _s3_mock_error("NoSuchBucket")
    result = load_prediction_history(
        bucket="bad", city="paris", target_name="bikes", station_id="s1", n_periods=5, s3_client=client
    )
    assert result.empty
    assert list(result.columns) == ["station_id", "ts", "score"]


# ── load_quality_recent ───────────────────────────────────────────────


def test_load_quality_recent_returns_rows_within_lookback():
    # Use a dt far in the past — should be filtered out
    old_dt = "2020-01-01-00-00"
    # Use a recent dt — should be kept (use a date we know is within 24h of "now")
    # We'll just use a very large lookback to avoid flakiness
    recent_dt = "2026-03-18-15-30"
    df_old = _quality_df(["s1"], old_dt)
    df_recent = _quality_df(["s2"], recent_dt)

    parquet_old = _make_parquet_bytes(df_old)
    parquet_recent = _make_parquet_bytes(df_recent)

    key_old = "monitoring/quality/target=bikes/city=paris/ds=2020-01-01/part-2020-01-01-00-00.parquet"
    key_recent = "monitoring/quality/target=bikes/city=paris/ds=2026-03-18/part-2026-03-18-15-30.parquet"

    client = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": [{"Key": key_old}, {"Key": key_recent}]}]
    client.get_paginator.return_value = paginator

    call_count = 0

    def _get_object(Bucket, Key):  # noqa: N803
        nonlocal call_count
        call_count += 1
        if Key == key_old:
            return {"Body": io.BytesIO(parquet_old)}
        return {"Body": io.BytesIO(parquet_recent)}

    client.get_object.side_effect = _get_object

    # Very large lookback — recent_dt row should survive the cutoff filter
    result = load_quality_recent(
        bucket="b", city="paris", target_name="bikes", s3_client=client, lookback_hours=999999
    )
    assert not result.empty
    assert set(result.columns) >= {"dt", "score", "label", "ts"}

    # With 0 lookback everything should be filtered out (all ts < now - 0h is impossible,
    # but any ts strictly before "now" will be excluded when lookback_hours=0)
    result_zero = load_quality_recent(
        bucket="b", city="paris", target_name="bikes", s3_client=client, lookback_hours=0
    )
    assert result_zero.empty


def test_load_quality_recent_bucket_not_found_returns_empty():
    client = _s3_mock_error("NoSuchBucket")
    result = load_quality_recent(bucket="bad", city="paris", target_name="bikes", s3_client=client)
    assert result.empty
    assert list(result.columns) == ["dt", "score", "label"]
