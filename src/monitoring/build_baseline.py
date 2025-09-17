# src/monitoring/build_baseline.py
# Purpose:
#   Build a DataQuality/DataDrift baseline for SageMaker Model Monitor from our
#   OFFLINE feature dataset (the same columns the endpoint receives online).
#
# Why features, not predictions?
#   - DataQuality/DataDrift monitor the INPUT distribution. Use our feature table.
#   - The Step 7 "monitoring/quality/..." join is for model-quality metrics (PR-AUC/F1),
#     not for the data baseline. See handler.py writes for monitoring/quality.  # (ref)
#
# How it works:
#   1) List recent feature Parquet partitions in S3 for the chosen city.
#   2) Load and concatenate a sample (keeping ONLY FEATURE_COLUMNS).
#   3) Write a local CSV sample (header=True).
#   4) Call SageMaker Model Monitor `suggest_baseline` to generate statistics/constraints
#      into s3://<bucket>/monitoring/baseline/city=<city>/...
#
# Tested on Windows (PowerShell), VS Code terminal.
#
# Example (PowerShell):
#   $env:AWS_REGION="ca-central-1"
#   $bucket = "mlops-bikeshare-387706002632-ca-central-1"
#   $roleArn = (aws iam get-role --role-name mlops-bikeshare-sagemaker-exec --query 'Role.Arn' --output text)
#   python -m src.monitoring.build_baseline `
#     --city nyc `
#     --bucket $bucket `
#     --role-arn $roleArn `
#     --days 7 `
#     --rows 200000
#
# Notes:
#   - You need `sagemaker`, `boto3`, `pandas`, `pyarrow`, `s3fs`.
#   - We import FEATURE_COLUMNS from our project schema to guarantee column parity.
#   - We drop non-feature columns so Model Monitor baseline matches the endpoint inputs.
#   - We avoid wildcards in pandas.read_parquet by listing keys and reading via pyarrow.

import argparse
import io
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from typing import List

import boto3
import pandas as pd
import pyarrow.parquet as pq
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.session import Session

# Reuse our project schema to ensure exact feature list
# (same import path used in our Step 7 code)  # (ref)
from src.features.schema import FEATURE_COLUMNS  # :contentReference[oaicite:2]{index=2}

# Optional: use our env loader if you prefer (not strictly required here)
try:
    from src.features.build_features import read_env  # provides bucket/region defaults if present
except Exception:
    read_env = None


def _s3(region_name: str = None):
    """Return a boto3 S3 client for the given region (or env default)."""
    session = boto3.session.Session(region_name=region_name)
    return session.client("s3")


def _list_feature_parquet_keys(s3, bucket: str, city: str, days: int) -> List[str]:
    """
    List S3 keys under features/city=<city>/dt=.../part-*.parquet for the last N days.
    We assume Hive-style partitions: features/city=nyc/dt=YYYY-MM-DD-HH-MM/part-*.parquet

    Returns a list of object keys sorted by dt (newest first).
    """
    prefix = f"features/city={city}/"
    paginator = s3.get_paginator("list_objects_v2")

    keys = []
    # We include a generous set and filter using a regex on dt=...
    dt_dir_re = re.compile(r"features/city=[^/]+/dt=(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})/")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if not k.endswith(".parquet"):
                continue
            m = dt_dir_re.search(k)
            if not m:
                continue
            # Parse the dt partition as UTC
            try:
                dt = datetime.strptime(m.group(1), "%Y-%m-%d-%H-%M").replace(tzinfo=timezone.utc)
            except Exception:
                continue

            if dt >= datetime.now(timezone.utc) - timedelta(days=days):
                keys.append((dt, k))

    # Newest first
    keys.sort(key=lambda x: x[0], reverse=True)
    return [k for _, k in keys]


def _read_parquet_files(s3, bucket: str, keys: List[str], max_rows: int) -> pd.DataFrame:
    """
    Download multiple small parquet files into memory and concatenate as a DataFrame.
    We stop when reaching ~max_rows to keep baseline small but representative.
    """
    frames = []
    rows = 0
    for k in keys:
        obj = s3.get_object(Bucket=bucket, Key=k)
        buf = io.BytesIO(obj["Body"].read())
        table = pq.read_table(buf)
        df = table.to_pandas(types_mapper=pd.ArrowDtype)  # preserve types

        frames.append(df)
        rows += len(df)
        if rows >= max_rows:
            break

    if not frames:
        raise RuntimeError("No parquet data found for the selected window.")
    return pd.concat(frames, ignore_index=True)


def _ensure_feature_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns the endpoint consumes.
    Your Step 7 handler passes FEATURE_COLUMNS into the endpoint payload, so we align here.  # (ref)
    """
    # handler.py builds payload with FEATURE_COLUMNS only.  # :contentReference[oaicite:3]{index=3}
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns in baseline source: {missing}")

    # IMPORTANT: Do NOT include label/target columns for data-quality baseline.
    # Keep exact inference inputs to avoid schema drift.
    return df[FEATURE_COLUMNS].copy()


def main():
    parser = argparse.ArgumentParser(description="Build SageMaker Model Monitor baseline from feature parquet.")
    parser.add_argument("--city", required=True, help="City partition, e.g., nyc")
    parser.add_argument("--bucket", required=False, help="S3 bucket name")
    parser.add_argument("--role_arn", required=True, help="SageMaker execution role ARN for the monitoring job")
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
        help="AWS region, e.g., ca-central-1",
    )
    parser.add_argument("--days", type=int, default=7, help="How many recent days of feature data to sample")
    parser.add_argument("--rows", type=int, default=200_000, help="Target maximum rows for the baseline sample")
    parser.add_argument("--baseline_prefix", default="monitoring/baseline", help="S3 prefix to write baseline outputs")
    args = parser.parse_args()

    # Resolve bucket/region defaults from our project env if not passed
    if read_env is not None and not args.bucket:
        cfg = read_env()
        # cfg is used throughout Step 7 to provide bucket/region/workgroup/db  # :contentReference[oaicite:4]{index=4}
        bucket = cfg["bucket"]
        region = args.region or cfg["region"]
    else:
        bucket = args.bucket
        region = args.region

    if not bucket:
        raise SystemExit("Bucket is required (pass --bucket or define via read_env).")

    s3 = _s3(region_name=region)

    print(
        f"[INFO] Listing feature parquet under s3://{bucket}/features/city={args.city}/ for last {args.days} days ..."
    )
    keys = _list_feature_parquet_keys(s3, bucket, args.city, args.days)
    if not keys:
        raise SystemExit("No parquet keys found. Check bucket, city, and days window.")

    print(f"[INFO] Will read up to {args.rows} rows from {len(keys)} objects (newest first).")
    df_raw = _read_parquet_files(s3, bucket, keys, max_rows=args.rows)

    print(f"[INFO] Raw sample shape: {df_raw.shape}")
    df_feat = _ensure_feature_only(df_raw)
    print(f"[INFO] Feature-only shape: {df_feat.shape}")

    # Write a local CSV (Windows-safe temp path)
    tmpdir = tempfile.gettempdir()
    baseline_file = os.path.join(tmpdir, f"baseline_{args.city}.csv")
    df_feat.to_csv(baseline_file, index=False)
    print(f"[INFO] Wrote local baseline CSV: {baseline_file}")

    sm_session = Session(boto_session=boto3.session.Session(region_name=region))

    # Prepare Model Monitor
    #   This will run a processing job under the given role that writes
    #   statistics.json + constraints.json to the output S3 URI.
    monitor = DefaultModelMonitor(
        role=args.role_arn,  # IAM role ARN or name
        instance_type="ml.m5.xlarge",  # default instance for the processing job
        instance_count=1,  # 1 is enough for baseline suggest
        volume_size_in_gb=30,  # adjust if our CSV is huge
        max_runtime_in_seconds=3600,  # give it time on first run
        sagemaker_session=sm_session,
    )

    output_uri = f"s3://{bucket}/{args.baseline_prefix}/city={args.city}/"
    print(f"[INFO] Suggesting baseline to: {output_uri}")

    monitor.suggest_baseline(
        baseline_dataset=f"s3://{bucket}/monitoring/baseline-inputs/baseline_nyc.csv",  # 直接用S3
        dataset_format={"csv": {"header": True}},
        output_s3_uri=f"s3://{bucket}/monitoring/baseline/city={args.city}/",
        wait=True,
    )

    print("[SUCCESS] Baseline generated. You should see statistics.json and constraints.json under:")
    print(f"{output_uri}")


if __name__ == "__main__":
    main()
