# -*- coding: utf-8 -*-
"""
Build hour-partitioned ModelQuality Ground Truth JSONL from a single DataCapture hour.

Given:
  - S3 bucket name (string)
  - City (e.g., "nyc")
  - One "capture hour" S3 prefix, e.g.
      s3://<bucket>/datacapture/endpoint=bikeshare-staging-.../bikeshare-staging/AllTraffic/2025/09/21/02/

What it does:
  1) List all capture files under that hour (both .jsonl and .jsonl.gz).
  2) Parse each line, extract the `inferenceId` from the capture record.
     (We assume you set InferenceId=f"{dt_local}_{station_id}" when invoking the endpoint.)
  3) For each `inferenceId`, split into dt_local (YYYY-MM-DD-HH-mm) and station_id.
  4) Load labels from your Step7 Parquet join:
       s3://<bucket>/monitoring/quality/city=<city>/ds=<YYYY-MM-DD>/part-*.parquet
     and get `y_stockout_bikes_30` for (dt_local, station_id).
  5) Write envelope JSONL to:
       s3://<bucket>/monitoring/quality/city=<city>/YYYY/MM/DD/HH/labels-YYYY-MM-DD-HH.jsonl
     where YYYY/MM/DD/HH are taken from the *capture hour prefix* (UTC).
This guarantees that merge finds the ground truth in the *same hour folder* as the capture.

Notes:
  - Lines with missing labels are skipped (printed as warnings).
  - Output is overwritten each run.
"""

import gzip
import json
import sys
from pathlib import PurePosixPath
from urllib.parse import urlparse

import boto3
import pandas as pd


def parse_s3_url(s3_url: str):
    u = urlparse(s3_url)
    if u.scheme != "s3":
        raise ValueError(f"Expect s3:// URL, got: {s3_url}")
    return u.netloc, u.path.lstrip("/")


def list_s3_keys(bucket: str, prefix: str):
    s3 = boto3.client("s3")
    keys = []
    cont = None
    while True:
        kw = dict(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
        if cont:
            kw["ContinuationToken"] = cont
        resp = s3.list_objects_v2(**kw)
        for item in resp.get("Contents", []):
            k = item["Key"]
            if k.endswith(".jsonl") or k.endswith(".jsonl.gz"):
                keys.append(k)
        if resp.get("IsTruncated"):
            cont = resp.get("NextContinuationToken")
        else:
            break
    return keys


def iter_capture_lines(bucket: str, key: str):
    """Yield JSON objects from a capture file (jsonl or jsonl.gz)."""
    s3 = boto3.client("s3")
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    if key.endswith(".gz"):
        data = gzip.decompress(body).decode("utf-8", errors="replace")
    else:
        data = body.decode("utf-8", errors="replace")
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            # drop bad lines silently, or log if needed
            continue


def extract_inference_id(rec: dict):
    """Return inferenceId from a capture record."""
    # Best-effort extraction across common capture formats
    md = rec.get("captureData", {}).get("eventMetadata", {})
    iid = md.get("inferenceId")
    if iid:
        return iid
    # Some formats put it under endpointOutput; keep as fallback
    iid = rec.get("captureData", {}).get("endpointOutput", {}).get("inferenceId")
    return iid


def write_jsonl_s3(lines, bucket, key):
    s3 = boto3.client("s3")
    body = ("\n".join(lines) + "\n").encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")


def run(bucket: str, city: str, capture_hour_prefix: str):
    # Parse capture prefix to get bucket/prefix and derive UTC YYYY/MM/DD/HH from it
    b2, p2 = parse_s3_url(capture_hour_prefix)
    if b2 != bucket:
        raise ValueError(f"Bucket mismatch: arg bucket={bucket}, prefix bucket={b2}")
    # Expect suffix .../<YYYY>/<MM>/<DD>/<HH>/
    path = PurePosixPath(p2)
    # Extract the 4 trailing parts
    try:
        HH = path.parts[-1] or path.parts[-1]  # if prefix ends with '/', last part could be ''
        if not HH:
            HH = path.parts[-2]
            DD = path.parts[-3]
            MM = path.parts[-4]
            YYYY = path.parts[-5]
        else:
            DD = path.parts[-2]
            MM = path.parts[-3]
            YYYY = path.parts[-4]
        # Normalize HH if the last part was '' (when prefix ends with '/')
        if len(HH) == 0 or HH == "":
            HH = path.parts[-2]
    except Exception as e:
        raise ValueError(f"Cannot parse YYYY/MM/DD/HH from: {capture_hour_prefix}") from e

    # 1) Collect inferenceIds from all files under this hour
    keys = list_s3_keys(bucket, p2)
    if not keys:
        print(f"[build-mq] No capture files under {capture_hour_prefix}")
        return

    inference_ids = set()
    for k in keys:
        for rec in iter_capture_lines(bucket, k):
            iid = extract_inference_id(rec)
            if iid:
                inference_ids.add(iid)

    if not inference_ids:
        print(f"[build-mq] No inferenceId found under {capture_hour_prefix}")
        return

    # 2) Build (dt_local, station_id) from inferenceId = "<YYYY-MM-DD-HH-mm>_<station_id>"
    pairs = []
    for iid in inference_ids:
        # split at the first '_' after the datetime block; your station_id does not contain underscores
        try:
            dt_local, station_id = iid.split("_", 1)
            pairs.append((iid, dt_local, station_id))
        except ValueError:
            # skip malformed ids
            continue

    # 3) Group by local date (ds) to read Step7 Parquet per day
    #    Parquet path: s3://<bucket>/monitoring/quality/city=<city>/ds=<YYYY-MM-DD>/
    #    Columns required: station_id, dt, y_stockout_bikes_30
    from collections import defaultdict

    by_ds = defaultdict(list)
    for iid, dt_local, station_id in pairs:
        ds = dt_local[:10]
        by_ds[ds].append((iid, dt_local, station_id))

    gt_lines = []
    for ds, items in by_ds.items():
        src = f"s3://{bucket}/monitoring/quality/city={city}/ds={ds}/"
        try:
            df = pd.read_parquet(src)
        except Exception as e:
            print(f"[build-mq] WARN: cannot read {src}: {e}")
            continue
        if df.empty:
            continue
        df = df[["station_id", "dt", "y_stockout_bikes_30"]].copy()
        df["dt"] = df["dt"].astype(str)
        # Build a quick lookup: (dt, station_id) -> label
        df["key"] = df["dt"] + "_" + df["station_id"]
        kv = dict(zip(df["key"], df["y_stockout_bikes_30"]))

        for iid, dt_local, station_id in items:
            label = kv.get(f"{dt_local}_{station_id}")
            if label is None:
                # No label for this inferenceId → skip (merge will simply ignore)
                print(f"[build-mq] WARN: missing label for {iid}")
                continue
            gt_lines.append(
                json.dumps(
                    {
                        "groundTruthData": {"data": str(int(float(label))), "encoding": "CSV"},
                        "eventMetadata": {"inferenceId": iid},
                        "eventVersion": "0",
                    },
                    ensure_ascii=False,
                )
            )

    if not gt_lines:
        print(f"[build-mq] No ground-truth lines produced for {capture_hour_prefix}")
        return

    out_key = f"monitoring/quality/city={city}/{YYYY}/{MM}/{DD}/{HH}/labels-{YYYY}-{MM}-{DD}-{HH}.jsonl"
    write_jsonl_s3(gt_lines, bucket, out_key)
    print(f"[build-mq] wrote {len(gt_lines)} lines → s3://{bucket}/{out_key}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python src/monitoring/build_mq_jsonl_from_capture_hour.py <bucket> <city> <s3-capture-hour-prefix>"
        )
        print("Example:")
        print(
            "  python src/monitoring/build_mq_jsonl_from_capture_hour.py "
            "mlops-bikeshare-... nyc "
            "s3://mlops-bikeshare-.../datacapture/.../AllTraffic/2025/09/21/02/"
        )
        sys.exit(2)
    run(sys.argv[1], sys.argv[2], sys.argv[3])
