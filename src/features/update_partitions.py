# src\features\update_partitions.py
# -*- coding: utf-8 -*-
"""
Incrementally add Glue/Athena partitions for the latest 4 five-minute windows.
Designed to replace expensive 'MSCK REPAIR TABLE' calls.


- As an AWS Lambda (recommended), triggered by EventBridge every 5 minutes.
- Locally for ad-hoc repair: `python -m src.features.update_partitions`

Environment variables (configure in Lambda console or PowerShell):
  REGION=ca-central-1
  ATHENA_WORKGROUP=primary
  ATHENA_OUTPUT=s3://<bucket>/athena_results/
  DB=mlops_bikeshare
  CITY=nyc
  BUCKET=mlops-bikeshare-...-ca-central-1

  # Table names in Glue/Athena:
  TBL_STATUS=station_status_raw
  TBL_INFO=station_information_raw
  TBL_WEATHER=weather_hourly_raw

  # S3 prefixes corresponding to each table (without leading/trailing slash):
  PREFIX_STATUS=raw/station_status
  PREFIX_INFO=raw/station_information
  PREFIX_WEATHER=raw/weather_hourly
"""

import datetime as dt
import json
import os
import time
from typing import List

import boto3


def _env(name: str, default: str = None) -> str:
    """Read environment variable with an optional default."""
    val = os.environ.get(name, default)
    if val is None or str(val).strip() == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def _floor_to_5min(ts: dt.datetime) -> dt.datetime:
    """Floor a UTC timestamp to the nearest lower 5-minute boundary."""
    # Ensure UTC and remove seconds/microseconds
    ts = ts.replace(second=0, microsecond=0, tzinfo=dt.timezone.utc)
    minute = (ts.minute // 5) * 5
    return ts.replace(minute=minute)


def _last_4_windows(now_utc: dt.datetime) -> List[str]:
    """
    Build the list of 4 window strings: 'YYYY-MM-DD-HH-mm' for
    [now_floor, now-5m, now-10m, now-15m].
    """
    base = _floor_to_5min(now_utc)
    dts = []
    for k in range(4):
        t = base - dt.timedelta(minutes=5 * k)
        dts.append(t.strftime("%Y-%m-%d-%H-%M"))
    return dts


def _alter_add_partition_sql(db: str, table: str, city: str, bucket: str, prefix: str, dt_str: str) -> str:
    """
    Build an ALTER TABLE ADD IF NOT EXISTS PARTITION statement with explicit LOCATION.
    This updates the Glue Catalog for the given (city, dt) only.
    """
    s3_loc = f"s3://{bucket}/{prefix}/city={city}/dt={dt_str}/"
    return (
        f"ALTER TABLE {db}.{table} "
        f"ADD IF NOT EXISTS PARTITION (city='{city}', dt='{dt_str}') "
        f"LOCATION '{s3_loc}'"
    )


def _athena_run(sql_batch: List[str], region: str, workgroup: str, output_s3: str, db: str) -> None:
    """
    Run a small batch of Athena DDL statements and wait for completion.
    Important: Provide QueryExecutionContext with Database (and Catalog) to
    satisfy Athena's requirement; some workgroups enforce this.
    """
    ath = boto3.client("athena", region_name=region)

    # Set explicit context (Database is required; Catalog is commonly AwsDataCatalog)
    qctx = {"Catalog": "AwsDataCatalog", "Database": db}

    for sql in sql_batch:
        resp = ath.start_query_execution(
            QueryString=sql,
            QueryExecutionContext=qctx,  # <-- the key fix
            WorkGroup=workgroup,
            ResultConfiguration={"OutputLocation": output_s3},
        )
        qid = resp["QueryExecutionId"]

        # Poll for completion
        while True:
            q = ath.get_query_execution(QueryExecutionId=qid)
            state = q["QueryExecution"]["Status"]["State"]
            if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
                break
            time.sleep(0.8)

        if state != "SUCCEEDED":
            reason = q["QueryExecution"]["Status"].get("StateChangeReason", "")
            raise RuntimeError(f"Athena DDL failed: state={state}; reason={reason}; sql={sql}")


def _build_sql_batch(
    city: str,
    db: str,
    bucket: str,
    tbl_status: str,
    tbl_info: str,
    tbl_weather: str,
    p_status: str,
    p_info: str,
    p_weather: str,
    windows: List[str],
) -> List[str]:
    """Create the ALTER TABLE statements for all tables and all windows."""
    batch = []
    for w in windows:
        batch.append(_alter_add_partition_sql(db, tbl_status, city, bucket, p_status, w))
        batch.append(_alter_add_partition_sql(db, tbl_info, city, bucket, p_info, w))
        batch.append(_alter_add_partition_sql(db, tbl_weather, city, bucket, p_weather, w))
    return batch


def run_once(now_utc: dt.datetime = None) -> dict:
    """Main logic that can be reused from Lambda handler or local CLI."""
    region = _env("REGION")
    workgroup = _env("ATHENA_WORKGROUP")
    output_s3 = _env("ATHENA_OUTPUT")
    db = _env("DB")  # <-- we will pass this down
    city = _env("CITY")
    bucket = _env("BUCKET")

    tbl_status = _env("TBL_STATUS", "station_status_raw")
    tbl_info = _env("TBL_INFO", "station_information_raw")
    tbl_weather = _env("TBL_WEATHER", "weather_hourly_raw")

    p_status = _env("PREFIX_STATUS", "raw/station_status")
    p_info = _env("PREFIX_INFO", "raw/station_information")
    p_weather = _env("PREFIX_WEATHER", "raw/weather_hourly")

    if now_utc is None:
        now_utc = dt.datetime.now(dt.timezone.utc)

    windows = _last_4_windows(now_utc)
    sql_batch = _build_sql_batch(
        city, db, bucket, tbl_status, tbl_info, tbl_weather, p_status, p_info, p_weather, windows
    )

    # Pass db here so QueryExecutionContext is not empty
    _athena_run(sql_batch, region=region, workgroup=workgroup, output_s3=output_s3, db=db)
    return {"added_or_ensured": [(city, w) for w in windows]}


# ---- Lambda entrypoint ----
def lambda_handler(event, context):
    """AWS Lambda handler entry. Returns a small JSON payload for logs."""
    result = run_once()
    return {"ok": True, **result}


# ---- Local CLI entrypoint ----
if __name__ == "__main__":
    # Allow local run for quick test
    out = run_once()
    print(json.dumps(out, indent=2))
