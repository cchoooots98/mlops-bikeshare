#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build Ground Truth JSONL per hour by rewriting rows from quality parquet.

- Input parquet location (per minute parts):
    s3://<bucket>/monitoring/quality/city=nyc/ds=YYYY-MM-DD/part-YYYY-MM-DD-HH-mm.parquet
- Output JSONL per hour:
    s3://<bucket>/monitoring/ground-truth/YYYY/MM/DD/HH/labels-YYYYMMDDHH.jsonl

Required columns in parquet:
    - inferenceId : str
    - y_stockout_bikes_30 : numeric (0/1)

Usage (Windows PowerShell friendly):
    python build_ground_truth_jsonl_simple.py --start 2025-09-29-06 --end 2025-09-29-08
"""

import argparse
import io
import json
import re
from datetime import datetime, timedelta
from typing import Dict, Iterator, List

import boto3
import pandas as pd
import pyarrow.parquet as pq

# -------------------- Configuration --------------------
BUCKET = "mlops-bikeshare-387706002632-ca-central-1"  # S3 bucket name
QUALITY_PREFIX = "monitoring/quality/city=nyc"  # where parquet lives
GROUNDTRUTH_PREFIX = "monitoring/ground-truth"  # where JSONL will be written

# Match keys like: .../ds=2025-09-29/part-2025-09-29-06-15.parquet
#   group(1): YYYY-MM-DD (day)
#   group(2): YYYY-MM-DD-HH (hour)
#   group(3): mm (minute)
PART_REGEX = re.compile(r".*/ds=(\d{4}-\d{2}-\d{2})/part-(\d{4}-\d{2}-\d{2}-\d{2})-(\d{2})\.parquet$")

s3 = boto3.client("s3")


# -------------------- Helpers --------------------
def parse_args():
    """Parse CLI arguments. Both start and end are inclusive, format YYYY-MM-DD-HH."""
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="Start timestamp inclusive, format YYYY-MM-DD-HH")
    p.add_argument("--end", required=True, help="End timestamp inclusive, format YYYY-MM-DD-HH")
    return p.parse_args()


def hour_range(start: datetime, end: datetime) -> List[datetime]:
    """Return a list of hourly datetimes from start..end inclusive."""
    out, cur = [], start
    while cur <= end:
        out.append(cur)
        cur += timedelta(hours=1)
    return out


def list_objects(prefix: str) -> Iterator[Dict]:
    """Yield S3 objects under a prefix (handles pagination)."""
    token = None
    while True:
        kwargs = {"Bucket": BUCKET, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            yield obj
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break


def extract_hour_from_key(key: str) -> datetime:
    """
    Parse hour from a parquet key.
    Example key: .../ds=2025-09-29/part-2025-09-29-06-15.parquet -> 2025-09-29 06:00 UTC (naive)
    """
    m = PART_REGEX.match(key)
    if not m:
        raise ValueError(f"Key does not match expected pattern: {key}")
    hour_str = m.group(2)  # YYYY-MM-DD-HH
    return datetime.strptime(hour_str, "%Y-%m-%d-%H")


def read_parquet_s3_to_df(key: str) -> pd.DataFrame:
    """Read a small parquet file from S3 to pandas."""
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    bio = io.BytesIO(obj["Body"].read())
    table = pq.read_table(bio)
    return table.to_pandas()


def write_jsonl_to_s3(key: str, lines: List[dict]) -> None:
    """Write a list of dicts as JSON Lines to S3."""
    buf = io.StringIO()
    for rec in lines:
        buf.write(json.dumps(rec, ensure_ascii=False))
        buf.write("\n")
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue().encode("utf-8"))


# -------------------- Core logic --------------------
def main():
    args = parse_args()
    start_dt = datetime.strptime(args.start, "%Y-%m-%d-%H")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d-%H")
    if end_dt < start_dt:
        raise ValueError("end must be >= start")

    # Pre-compute target hours and day partitions we need to look at.
    hours = hour_range(start_dt, end_dt)
    ds_days = sorted({h.strftime("%Y-%m-%d") for h in hours})

    # For each day partition, collect parquet parts that fall into the requested hours.
    keys_by_hour: Dict[datetime, List[str]] = {h: [] for h in hours}
    for ds in ds_days:
        prefix = f"{QUALITY_PREFIX}/ds={ds}/"
        for obj in list_objects(prefix):
            key = obj["Key"]
            try:
                hour_dt = extract_hour_from_key(key)
            except ValueError:
                # Skip unexpected file names defensively
                continue
            if start_dt <= hour_dt <= end_dt and key.endswith(".parquet"):
                keys_by_hour[hour_dt].append(key)

    # For each hour, read all minute parts, then write one JSONL file.
    for hour_dt in hours:
        parts = sorted(keys_by_hour.get(hour_dt, []))
        if not parts:
            print(f"[{hour_dt:%Y-%m-%d %H}:00] No parquet parts found. Skip.")
            continue

        frames = []
        for key in parts:
            df = read_parquet_s3_to_df(key)

            # Validate required columns exist
            missing = [c for c in ("inferenceId", "y_stockout_bikes_30") if c not in df.columns]
            if missing:
                raise KeyError(f"Missing columns {missing} in {key}")

            # Keep only needed columns
            frames.append(df[["inferenceId", "y_stockout_bikes_30"]])

        if not frames:
            print(f"[{hour_dt:%Y-%m-%d %H}:00] No rows after filtering. Skip.")
            continue

        joined = pd.concat(frames, ignore_index=True)
        joined = joined.dropna(subset=["inferenceId", "y_stockout_bikes_30"])

        # Build JSONL lines (eventId = inferenceId; no capture alignment)
        lines: List[dict] = []
        for rec in joined.itertuples(index=False):
            inference_id = str(getattr(rec, "inferenceId"))
            # Normalize label to int 0/1
            try:
                label_int = int(float(getattr(rec, "y_stockout_bikes_30")))
            except Exception:
                # Skip rows with invalid labels
                continue

            lines.append(
                {
                    "groundTruthData": {"data": str(label_int), "encoding": "CSV"},
                    "eventMetadata": {"eventId": inference_id},  # directly use inferenceId
                    "eventVersion": "0",
                }
            )

        if not lines:
            print(f"[{hour_dt:%Y-%m-%d %H}:00] No valid rows to write. Skip.")
            continue

        # Output path: monitoring/ground-truth/YYYY/MM/DD/HH/labels-YYYYMMDDHH.jsonl
        y, m, d, h = hour_dt.strftime("%Y"), hour_dt.strftime("%m"), hour_dt.strftime("%d"), hour_dt.strftime("%H")
        out_key = f"{GROUNDTRUTH_PREFIX}/{y}/{m}/{d}/{h}/labels-{hour_dt:%Y%m%d%H}.jsonl"
        write_jsonl_to_s3(out_key, lines)

        print(f"[{hour_dt:%Y-%m-%d %H}:00] Wrote {len(lines)} labels -> s3://{BUCKET}/{out_key}")

    print("Done.")


if __name__ == "__main__":
    main()
