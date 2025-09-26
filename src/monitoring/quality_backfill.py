# src/monitoring/quality_backfill.py
# Windows/VSCode friendly. All comments in English.
# Purpose:
#   - Backfill model-quality shards for matured prediction timepoints (dt+30 min passed).
#   - Look back a sliding window (default 120 minutes, 5-min grid).
#   - Only process dts that ALREADY HAVE predictions parquet.
#   - Write one parquet shard per dt:
#       s3://{bucket}/monitoring/quality/city={city}/ds=YYYY-MM-DD/part-{dt}.parquet

import io
from datetime import datetime, timedelta, timezone
from typing import List

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.features.build_features import athena_conn, query_df, read_env

# ---------- Tunables ----------

MATURITY_MINUTES = 30  # t+30m actuals
BACKFILL_MINUTES = 120  # lookback horizon
STEP_MINUTES = 5  # 5-min grid
STOCKOUT_THRESHOLD = 2  # bikes <= 2 => stockout=1.0


# ---------- AWS clients ----------


def _s3():
    return boto3.client("s3")


# ---------- S3 helpers ----------


def _object_exists(bucket: str, key: str) -> bool:
    """Check if an S3 object exists without downloading it."""
    try:
        _s3().head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def _read_parquet_s3(bucket: str, key: str) -> pd.DataFrame:
    """Read a Parquet object from S3 into a pandas DataFrame."""
    try:
        obj = _s3().get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
        table = pq.read_table(io.BytesIO(data))
        return table.to_pandas()
    except Exception:
        return pd.DataFrame()


def _write_parquet_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    """Write a small dataframe to S3 as a single Parquet object."""
    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    _s3().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


# ---------- Key helpers ----------


def _prediction_key(city: str, dt: str) -> str:
    return f"inference/city={city}/dt={dt}/predictions.parquet"


def _quality_key(city: str, dt: str) -> str:
    ds = dt[:10]  # YYYY-MM-DD
    return f"monitoring/quality/city={city}/ds={ds}/part-{dt}.parquet"


# ---------- Backfill time grid ----------


def _make_candidate_dts(latest_dt_str: str, now_utc: datetime) -> List[str]:
    """
    Build a 5-min grid backwards from latest_dt_str for BACKFILL_MINUTES,
    keep only matured dts (now >= dt + 30m).
    """
    base = datetime.strptime(latest_dt_str, "%Y-%m-%d-%H-%M").replace(tzinfo=timezone.utc)
    out = []
    steps = BACKFILL_MINUTES // STEP_MINUTES
    for k in range(0, steps + 1):
        dtk = base - timedelta(minutes=STEP_MINUTES * k)
        if now_utc >= (dtk + timedelta(minutes=MATURITY_MINUTES)):
            out.append(dtk.strftime("%Y-%m-%d-%H-%M"))
    return out


# ---------- Actuals ----------


def _compute_actuals_for_dt(cnx, city: str, pred_dt: str, threshold: int = STOCKOUT_THRESHOLD) -> pd.DataFrame:
    """
    Look up bikes(t+30m) and produce the binary label:
    y_stockout_bikes_30 = 1.0 if bikes_t30 <= threshold else 0.0
    """
    dt_plus30 = (datetime.strptime(pred_dt, "%Y-%m-%d-%H-%M") + timedelta(minutes=30)).strftime("%Y-%m-%d-%H-%M")
    sql = f"""
    SELECT station_id, bikes AS bikes_t30
    FROM {cnx.schema_name}.v_station_status
    WHERE city = '{city}' AND dt = '{dt_plus30}'
    """
    df = query_df(cnx, sql)
    if df.empty:
        return pd.DataFrame(columns=["station_id", "bikes_t30", "y_stockout_bikes_30", "dt_plus30"])

    df["y_stockout_bikes_30"] = (df["bikes_t30"] <= threshold).astype("float64")
    df["dt_plus30"] = dt_plus30
    return df


# ---------- Main entry ----------


def main():
    # 1) Config + Athena connection
    cfg = read_env()
    city = cfg["city"]
    bucket = cfg["bucket"]

    cnx = athena_conn(
        region=cfg["region"],
        s3_staging_dir=cfg["athena_output"],
        workgroup=cfg["athena_workgroup"],
        schema_name=cfg["athena_database"],
    )

    # 2) Determine the latest dt in predictions (use the newest partition we have)
    #    This avoids scanning raw tables again and keeps the job lightweight.
    #    If you have an "inference" external table, prefer querying it:
    try:
        df = pd.read_sql(
            f"SELECT max(dt) as latest_dt FROM {cnx.schema_name}.inference WHERE city='{city}'",
            cnx,
        )
        latest_dt = str(df.loc[0, "latest_dt"])
        if not latest_dt or latest_dt == "None":
            print("[quality] no predictions found; nothing to backfill")
            return
    except Exception:
        print("[quality] failed to read latest dt from 'inference' table; consider running MSCK REPAIR TABLE inference")
        return

    # 3) Build matured candidate dts and iterate
    now_utc = datetime.now(timezone.utc)
    candidates = _make_candidate_dts(latest_dt, now_utc)

    shards_written = 0
    for dt_pred in candidates:
        pred_key = _prediction_key(city, dt_pred)
        qual_key = _quality_key(city, dt_pred)

        # Skip if we already have the quality shard (idempotent)
        if _object_exists(bucket, qual_key):
            continue

        # Skip if the predictions parquet is missing (we never predict-retroactively here)
        if not _object_exists(bucket, pred_key):
            continue

        preds = _read_parquet_s3(bucket, pred_key)
        if preds.empty:
            continue

        # Compute actuals at t+30m; skip if not matured yet
        acts = _compute_actuals_for_dt(cnx, city, dt_pred, threshold=STOCKOUT_THRESHOLD)
        if acts.empty:
            continue

        # Merge on station_id; keep inferenceId for traceability
        joined = (
            preds.merge(
                acts[["station_id", "bikes_t30", "y_stockout_bikes_30", "dt_plus30"]],
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
            continue

        _write_parquet_s3(joined, bucket, qual_key)
        shards_written += 1

    # 4) Best-effort partition repair for the monitoring table
    try:
        pd.read_sql("MSCK REPAIR TABLE monitoring_quality", cnx)
    except Exception:
        pass

    print(f"[quality] wrote {shards_written} shard(s).")


if __name__ == "__main__":
    # PowerShell:
    # PS> python -m src.monitoring.quality_backfill
    main()
