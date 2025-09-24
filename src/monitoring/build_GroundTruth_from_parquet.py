#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build Model Quality Ground Truth JSONL from prediction+actuals parquet files.

Input (per-row parquet schema must include at least):
  - inferenceId : str
  - y_stockout_bikes_30 : numeric (0.0/1.0)

Behavior:
  * Walk S3 keys like:
      s3://mlops-bikeshare-387706002632-ca-central-1/monitoring/quality/city=nyc/ds=YYYY-MM-DD/part-YYYY-MM-DD-HH-mm.parquet
  * Filter by [start_ts, end_ts] inclusive where ts format is YYYY-MM-DD-HH
  * Group records by hour, write JSONL to:
      s3://mlops-bikeshare-387706002632-ca-central-1/monitoring/ground-truth/Y/M/D/H/labels-YYYYMMDDHH.jsonl

Usage:
  python build_ground_truth_jsonl.py --start 2025-09-22-21 --end 2025-09-22-23
"""

import argparse
import io
import json
import re
from datetime import datetime, timedelta
from typing import Dict, Iterator, List, Tuple

import boto3
import pandas as pd
import pyarrow.parquet as pq

# ---- Configuration constants ----
BUCKET = "mlops-bikeshare-387706002632-ca-central-1"
QUALITY_PREFIX = "monitoring/quality/city=nyc"  # where parquet lives
GROUNDTRUTH_PREFIX = "monitoring/ground-truth"  # where JSONL will be written

PART_REGEX = re.compile(r".*/ds=(\d{4}-\d{2}-\d{2})/part-(\d{4}-\d{2}-\d{2}-\d{2})-(\d{2})\.parquet$")

s3 = boto3.client("s3")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="Start timestamp inclusive, format YYYY-MM-DD-HH")
    p.add_argument("--end", required=True, help="End timestamp inclusive, format YYYY-MM-DD-HH")
    p.add_argument("--endpoint", required=True, help="Endpoint name")
    return p.parse_args()


def hour_range(start: datetime, end: datetime) -> List[datetime]:
    """Return a list of hourly datetimes from start..end inclusive."""
    out = []
    cur = start
    while cur <= end + timedelta(hours=1):
        out.append(cur)
        cur = cur + timedelta(hours=1)
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


def extract_ds_hour_from_key(key: str) -> Tuple[datetime, str]:
    """
    From a key like .../ds=2025-09-22/part-2025-09-22-21-55.parquet
    return (hour_dt, minute) where hour_dt=2025-09-22-21:00 and minute='55'.
    Raise ValueError if it doesn't match expected pattern.
    """
    m = PART_REGEX.match(key)
    if not m:
        raise ValueError(f"Key does not match expected pattern: {key}")
    _, hour_str, minute = m.group(1), m.group(2), m.group(3)
    # hour_str format: YYYY-MM-DD-HH
    hour_dt = datetime.strptime(hour_str, "%Y-%m-%d-%H")
    return hour_dt, minute


def read_parquet_s3_to_df(bucket: str, key: str) -> pd.DataFrame:
    """Read parquet from S3 into a pandas DataFrame (memory-friendly for small files)."""
    obj = s3.get_object(Bucket=bucket, Key=key)
    bio = io.BytesIO(obj["Body"].read())
    table = pq.read_table(bio)
    return table.to_pandas()


def write_jsonl_to_s3(bucket: str, key: str, lines: List[dict]):
    """Write a list of JSON serializable dicts as JSONL to S3."""
    buf = io.StringIO()
    for rec in lines:
        buf.write(json.dumps(rec, ensure_ascii=False))
        buf.write("\n")
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue().encode("utf-8"))


def get_endpoint_config(endpoint_name: str) -> str:
    """
    Get the endpoint config name for a given endpoint name using boto3.
    """
    sm = boto3.client("sagemaker")
    resp = sm.describe_endpoint(EndpointName=endpoint_name)
    return resp["EndpointConfigName"]


def list_jsonl_files(start: datetime, end: datetime, endpoint_config: str, endpoint_name: str) -> List[str]:
    """List all .jsonl files in the specified S3 path within the time range."""
    start_dt = start - timedelta(hours=1)
    end_dt = end + timedelta(hours=1)
    jsonl_files = []

    # Generate the S3 prefix for the search
    for hour in hour_range(start_dt, end_dt):
        y = hour.strftime("%Y")
        m = hour.strftime("%m")
        d = hour.strftime("%d")
        h = hour.strftime("%H")
        prefix = f"datacapture/endpoint={endpoint_config}/{endpoint_name}/AllTraffic/{y}/{m}/{d}/{h}/"

        for obj in list_objects(prefix):
            if obj["Key"].endswith(".jsonl"):
                jsonl_files.append(obj["Key"])

    return jsonl_files


def extract_event_metadata_from_jsonl(bucket: str, jsonl_files: List[str]) -> Dict[str, Dict]:
    """
    Extract event metadata from model data capture JSONL files.
    Key by inferenceId -> {eventId, inferenceTime}.
    If a JSON line does not contain the needed fields, skip it safely.
    NOTE:
    - Some data-capture records may not include eventMetadata.inferenceId.
      Model Monitor aligns ground-truth with data-capture by eventId, so we only
      keep entries where BOTH inferenceId and eventId are present.
    """
    event_metadata: Dict[str, Dict] = {}
    total_lines = 0
    kept = 0
    skipped_no_md = 0
    skipped_no_inf_id = 0
    skipped_no_event_id = 0

    for key in jsonl_files:
        obj = s3.get_object(Bucket=bucket, Key=key)
        lines = obj["Body"].read().decode("utf-8").splitlines()
        for line in lines:
            total_lines += 1
            try:
                record = json.loads(line)
                md = record.get("eventMetadata", None)
                if not md:
                    skipped_no_md += 1
                    continue

                inf_id = md.get("inferenceId", None)
                if not inf_id:
                    # If inferenceId is absent, we cannot map parquet inferenceId -> eventId; skip.
                    skipped_no_inf_id += 1
                    continue

                ev_id = md.get("eventId", None)
                if not ev_id:
                    # Model Quality requires eventId to join with data capture; skip.
                    skipped_no_event_id += 1
                    continue

                event_metadata[str(inf_id)] = {
                    "eventId": str(ev_id),
                    "inferenceTime": md.get("inferenceTime", ""),
                }
                kept += 1
            except Exception:
                # Ignore malformed lines defensively
                continue
    return event_metadata


def build_ground_truth_record_with_metadata(inferenceId: str, label_val, event_metadata: Dict) -> dict:
    """
    Build one ground-truth JSONL line.
    REQUIREMENT: event_metadata must have a mapping for this inferenceId (providing eventId).
    If the mapping is missing, the caller should SKIP this record (do not write an empty eventId).
    """
    # Normalize label to int {0,1}
    try:
        label_int = int(float(label_val))
    except Exception:
        raise ValueError(f"Invalid label value: {label_val!r}")

    # Lookup capture metadata by inferenceId
    event_info = event_metadata.get(str(inferenceId))
    if not event_info or not event_info.get("eventId"):
        # The caller will skip this record; we keep this function pure.
        raise KeyError("Missing eventId mapping for this inferenceId")

    return {
        "groundTruthData": {"data": str(label_int), "encoding": "CSV"},
        "eventMetadata": {
            "eventId": event_info["eventId"],  # MUST be present for Model Quality join
            "inferenceId": str(inferenceId),  # Optional, helpful for debugging
            "inferenceTime": event_info.get("inferenceTime", ""),
        },
        "eventVersion": "0",
    }


def main():
    args = parse_args()
    start_dt = datetime.strptime(args.start, "%Y-%m-%d-%H")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d-%H")

    if end_dt < start_dt:
        raise ValueError("end must be >= start")

    # Get endpoint config
    endpoint_name = args.endpoint
    endpoint_config = get_endpoint_config(endpoint_name)

    # List .jsonl files
    jsonl_files = list_jsonl_files(start_dt, end_dt, endpoint_config, endpoint_name)

    # Extract event metadata
    event_metadata = extract_event_metadata_from_jsonl(BUCKET, jsonl_files)

    # Pre-compute all hours we need to cover
    hours = set(hour_range(start_dt, end_dt))

    # List only ds partitions that might intersect the time window
    # (ds=YYYY-MM-DD is driven by the hour's date)
    ds_days = sorted({h.strftime("%Y-%m-%d") for h in hours})

    # Collect candidate parquet keys within day partitions and keep those within window
    keys_by_hour: Dict[datetime, List[str]] = {h: [] for h in hours}

    for ds in ds_days:
        prefix = f"{QUALITY_PREFIX}/ds={ds}/"
        for obj in list_objects(prefix):
            key = obj["Key"]
            try:
                hour_dt, _minute = extract_ds_hour_from_key(key)
            except ValueError:
                # Skip any unexpected files
                continue
            if start_dt <= hour_dt <= end_dt:
                keys_by_hour[hour_dt].append(key)

    # For each hour, read all parquet fragments, build JSONL lines, and write to the target prefix
    for hour_dt in sorted(hours):
        parts = keys_by_hour.get(hour_dt, [])
        if not parts:
            # No data captured for this hour; skip
            continue

        # Concatenate all minute-part parquet files within the same hour
        frames = []
        for key in sorted(parts):
            df = read_parquet_s3_to_df(BUCKET, key)

            # Basic schema validation (fail fast if required columns missing)
            missing = [c for c in ("inferenceId", "y_stockout_bikes_30") if c not in df.columns]
            if missing:
                raise KeyError(f"Missing columns {missing} in {key}")

            # Keep only the columns necessary to build ground truth
            frames.append(df[["inferenceId", "y_stockout_bikes_30"]])

        if not frames:
            continue

        joined = pd.concat(frames, ignore_index=True)
        # Drop rows with NaN inferenceId or label
        joined = joined.dropna(subset=["inferenceId", "y_stockout_bikes_30"])

        # Convert to JSONL lines
        lines = []
        skipped_no_mapping = 0
        for rec in joined.itertuples(index=False):
            inferenceId = getattr(rec, "inferenceId")
            label_val = getattr(rec, "y_stockout_bikes_30")
            try:
                line = build_ground_truth_record_with_metadata(inferenceId, label_val, event_metadata)
                lines.append(line)
            except KeyError:
                # No eventId mapping for this inferenceId; skip
                skipped_no_mapping += 1
                continue

        if not lines:
            print(f"[{hour_dt}] No valid records with eventId mapping; skip writing.")
            continue

        # Compute destination key: monitoring/ground-truth/Y/M/D/H/labels-YYYYMMDDHH.jsonl
        y = hour_dt.strftime("%Y")
        m = hour_dt.strftime("%m")
        d = hour_dt.strftime("%d")
        h = hour_dt.strftime("%H")
        out_prefix = f"{GROUNDTRUTH_PREFIX}/{y}/{m}/{d}/{h}/"
        out_key = f"{out_prefix}labels-{hour_dt.strftime('%Y%m%d%H')}.jsonl"

        write_jsonl_to_s3(BUCKET, out_key, lines)
        print(f"Wrote {len(lines)} labels -> s3://{BUCKET}/{out_key}")

    print("Done.")


if __name__ == "__main__":
    main()
