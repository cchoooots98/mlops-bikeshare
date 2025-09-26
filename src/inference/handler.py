# handler.py (improved)
# End-to-end:
# 1) Build online features and invoke the SageMaker endpoint (row-wise with InferenceId).
# 2) Write predictions to s3://.../inference/city=.../dt=.../predictions.parquet
# 3) Backfill model-quality for any matured prediction dt within a sliding window.
#    - Window length is configurable (default 120 minutes).
#    - 5-minute grid to align with raw data cadence.
#    - Only backfill for dt that actually has predictions.
#    - Idempotent: skip if quality shard already exists.
#    - Reads the correct predictions for each dt (no reuse of "latest" predictions).
#
# Windows/PowerShell ready.

import io
import json
import os
import warnings
from datetime import datetime, timedelta, timezone
from typing import List

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.features.build_features import athena_conn, query_df, read_env
from src.features.schema import FEATURE_COLUMNS
from src.inference.featurize_online import build_online_features

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

# ===== Tunables =====
YHAT_PROB_THRESHOLD = 0.15  # Decision threshold for binary label from probability
STOCKOUT_THRESHOLD = 2  # bikes <= 2 => y=1 within 30 minutes
MATURITY_MINUTES = 30  # Ground-truth maturity (t+30m)
BACKFILL_MINUTES = 120  # Backfill window length
STEP_MINUTES = 5  # 5-min grid to align with data ingestion
MAX_CANDIDATES = BACKFILL_MINUTES // STEP_MINUTES  # Safety cap (e.g., 24 for 120m/5m)

# ==================== Clients ====================


def _s3():
    return boto3.client("s3")


def _smr():
    return boto3.client("sagemaker-runtime")


# ==================== S3 Utils ====================


def _write_parquet_s3(df: pd.DataFrame, bucket: str, key: str):
    """Write a small DataFrame to S3 as Parquet using an in-memory buffer (Windows friendly)."""
    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    _s3().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


def _read_parquet_s3(bucket: str, key: str) -> pd.DataFrame:
    """Read a Parquet object from S3 into a pandas DataFrame (no temp files)."""
    try:
        obj = _s3().get_object(Bucket=bucket, Key=key)
    except _s3().exceptions.NoSuchKey:
        return pd.DataFrame()
    data = obj["Body"].read()
    table = pq.read_table(io.BytesIO(data))
    return table.to_pandas()


def _object_exists(bucket: str, key: str) -> bool:
    """Check if an S3 object exists without downloading it."""
    try:
        _s3().head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def _prefix_has_any_object(bucket: str, prefix: str) -> bool:
    """Check if there is any object under a prefix (used to test 'already backfilled?')."""
    resp = _s3().list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return "Contents" in resp


# ==================== Athena Tables (idempotent) ====================


def _inference_table_create_if_absent(cnx, bucket: str):
    """Create the external table for predictions if missing."""
    sql = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS inference (
        station_id string,
        yhat_bikes double,
        yhat_bikes_bin double,
        inferenceId string,
        raw string
    )
    PARTITIONED BY (`city` string, `dt` string)
    STORED AS PARQUET
    LOCATION 's3://{bucket}/inference/'
    TBLPROPERTIES ('parquet.compression'='SNAPPY')
    """
    pd.read_sql(sql, cnx)
    pd.read_sql("MSCK REPAIR TABLE inference", cnx)


def _quality_table_create_if_absent(cnx, bucket: str):
    """Create the external table for monitoring quality if missing."""
    sql = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS monitoring_quality (
        station_id string,
        dt string,
        dt_plus30 string,
        yhat_bikes double,
        yhat_bikes_bin double,
        y_stockout_bikes_30 double,
        bikes_t30 int,
        inferenceId string
    )
    PARTITIONED BY (city string, ds string)
    STORED AS PARQUET
    LOCATION 's3://{bucket}/monitoring/quality/'
    TBLPROPERTIES ('parquet.compression'='SNAPPY')
    """
    pd.read_sql(sql, cnx)
    pd.read_sql("MSCK REPAIR TABLE monitoring_quality", cnx)


# ==================== Inference & Actuals ====================


def _invoke_endpoint_rowwise(endpoint_name: str, X: pd.DataFrame) -> pd.DataFrame:
    """
    Invoke the SageMaker endpoint one record at a time with a deterministic InferenceId.
    This ensures Ground Truth can be joined later.
    """
    rows = []
    rt = _smr()

    for rec in X[["city", "dt", "station_id"] + FEATURE_COLUMNS].itertuples(index=False, name=None):
        city, dt_str, station_id, *features = rec

        payload = {
            "inputs": {
                "dataframe_split": {
                    "columns": FEATURE_COLUMNS,
                    "data": [list(map(float, features))],
                }
            }
        }

        # Deterministic inference id: dt + station_id
        inferenceId = f"{dt_str}_{station_id}"

        resp = rt.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            InferenceId=inferenceId,
            Body=json.dumps(payload).encode("utf-8"),
        )
        body = resp["Body"].read()
        try:
            out = json.loads(body.decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"Bad model response for {inferenceId}: {body[:500]}") from e

        def to_scalar(x):
            # Normalize common response shapes to a float
            if isinstance(x, (int, float)):
                return float(x)
            if isinstance(x, list) and len(x) == 1 and isinstance(x[0], (int, float)):
                return float(x[0])
            if isinstance(x, dict) and "yhat" in x:
                return float(x["yhat"])
            return float("nan")

        preds = out["predictions"] if isinstance(out, dict) and "predictions" in out else out
        yhat = to_scalar(preds[0] if isinstance(preds, list) else preds)

        rows.append(
            {
                "city": city,
                "dt": dt_str,
                "station_id": station_id,
                "yhat_bikes": yhat,
                "yhat_bikes_bin": float(yhat >= YHAT_PROB_THRESHOLD),
                "inferenceId": inferenceId,
                "raw": json.dumps(out, ensure_ascii=False),
            }
        )

    return pd.DataFrame(rows)


def _compute_actuals_for_dt(cnx, city: str, pred_dt: str, threshold: int = STOCKOUT_THRESHOLD) -> pd.DataFrame:
    """
    Build t+30m actuals from v_station_status, then compute label:
    y_stockout_bikes_30 = 1.0 if bikes(t+30m) <= threshold else 0.0
    """
    dt_plus30 = (datetime.strptime(pred_dt, "%Y-%m-%d-%H-%M") + timedelta(minutes=30)).strftime("%Y-%m-%d-%H-%M")
    sql = f"""
    SELECT station_id, bikes AS bikes_t30
    FROM {cnx.schema_name}.v_station_status
    WHERE city = '{city}' AND dt = '{dt_plus30}'
    """
    df = query_df(cnx, sql)
    if df.empty:
        # Not matured yet or missing raw snapshot; skip for now.
        return pd.DataFrame(columns=["station_id", "bikes_t30", "y_stockout_bikes_30", "dt_plus30"])

    df["y_stockout_bikes_30"] = (df["bikes_t30"] <= threshold).astype("float64")
    df["dt_plus30"] = dt_plus30
    return df


# ==================== Backfill Helpers ====================


def _make_candidate_dts(latest_dt_str: str, now_utc: datetime) -> List[str]:
    """
    Build a 5-minute grid backwards from latest_dt_str for BACKFILL_MINUTES,
    keeping only matured dts (now >= dt + MATURITY_MINUTES).
    """
    base = datetime.strptime(latest_dt_str, "%Y-%m-%d-%H-%M").replace(tzinfo=timezone.utc)
    out = []
    for k in range(0, MAX_CANDIDATES + 1):
        dtk = base - timedelta(minutes=STEP_MINUTES * k)
        if now_utc >= (dtk + timedelta(minutes=MATURITY_MINUTES)):
            out.append(dtk.strftime("%Y-%m-%d-%H-%M"))
    return out


def _prediction_key(city: str, dt: str) -> str:
    """S3 key for predictions written by this job."""
    return f"inference/city={city}/dt={dt}/predictions.parquet"


def _quality_key(city: str, dt: str) -> str:
    """S3 key prefix for monitoring quality for this pred dt."""
    ds = dt[:10]  # YYYY-MM-DD
    # Using part-{dt}.parquet as you do; if you sharded later, keep the prefix check.
    return f"monitoring/quality/city={city}/ds={ds}/part-{dt}.parquet"


def _quality_prefix_for_day(city: str, dt: str) -> str:
    """Day-level prefix helper (not strictly needed but handy)."""
    ds = dt[:10]
    return f"monitoring/quality/city={city}/ds={ds}/"


# ==================== Main ====================


def main():
    # ----- 1) Config & Connections -----
    cfg = read_env()
    city = cfg["city"]
    bucket = cfg["bucket"]
    endpoint_name = os.environ.get("SM_ENDPOINT", "bikeshare-staging")

    cnx = athena_conn(
        region=cfg["region"],
        s3_staging_dir=cfg["athena_output"],
        workgroup=cfg["athena_workgroup"],
        schema_name=cfg["athena_database"],
    )

    _inference_table_create_if_absent(cnx, bucket)
    _quality_table_create_if_absent(cnx, bucket)

    # ----- 2) Inference for the latest snapshot -----
    X = build_online_features(city)  # ["city","dt","station_id"] + FEATURE_COLUMNS, sorted with latest first
    latest_dt = X["dt"].iloc[0]

    preds_latest = _invoke_endpoint_rowwise(endpoint_name, X)

    pred_key_latest = _prediction_key(city, latest_dt)
    _write_parquet_s3(
        preds_latest[["station_id", "dt", "yhat_bikes", "yhat_bikes_bin", "inferenceId", "raw"]],
        bucket,
        pred_key_latest,
    )

    # Lightweight partition repair (best-effort)
    try:
        pd.read_sql("MSCK REPAIR TABLE inference", cnx)
    except Exception:
        pass

    # ----- 3) Backfill loop (windowed, matured, idempotent) -----
    now_utc = datetime.now(timezone.utc)
    candidates = _make_candidate_dts(latest_dt, now_utc)

    processed = 0
    for dt_pred in candidates:
        if processed >= MAX_CANDIDATES:
            break  # Safety cap

        # Skip if quality shard already exists (idempotent)
        qual_key = _quality_key(city, dt_pred)
        if _object_exists(bucket, qual_key):
            continue

        # Skip if predictions for this dt do not exist (no point in backfilling)
        pred_key = _prediction_key(city, dt_pred)
        if not _object_exists(bucket, pred_key):
            continue

        # Read predictions for THIS dt (very important: do not reuse "latest" batch)
        preds_dt = _read_parquet_s3(bucket, pred_key)
        if preds_dt.empty:
            continue

        # Compute actuals (t+30) for THIS dt
        actuals = _compute_actuals_for_dt(cnx, city, dt_pred, threshold=STOCKOUT_THRESHOLD)
        if actuals.empty:
            # Not matured yet or missing raw snapshot; skip and let a later run fill it
            continue

        # Join on station_id; preserve inferenceId for traceability
        joined = (
            preds_dt.merge(
                actuals[["station_id", "bikes_t30", "y_stockout_bikes_30", "dt_plus30"]],
                on="station_id",
                how="inner",
            )
            .assign(dt=lambda d: dt_pred)
            .loc[
                :,
                [
                    "station_id",
                    "dt",
                    "dt_plus30",
                    "yhat_bikes",
                    "yhat_bikes_bin",
                    "y_stockout_bikes_30",
                    "bikes_t30",
                    "inferenceId",
                ],
            ]
        )

        if joined.empty:
            # Unlikely, but keep it safe
            continue

        # Write a single shard per pred dt (idempotent filename)
        _write_parquet_s3(joined, bucket, qual_key)

        # Best-effort partition repair for the day
        try:
            pd.read_sql("MSCK REPAIR TABLE monitoring_quality", cnx)
        except Exception:
            pass

        processed += 1


if __name__ == "__main__":
    # PowerShell example:
    # PS> $env:SM_ENDPOINT="bikeshare-prod"; python src/inference/handler.py
    main()
