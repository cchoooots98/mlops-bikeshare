# capture_to_batchjsonl.py
# Purpose:
#   1) Discover the S3 capture prefix used by a SageMaker endpoint via its endpoint-config.
#   2) For a given UTC date (YYYY-MM-DD or YYYY/MM/DD), read all capture files under each hour
#      datacapture/.../<endpoint-name>/AllTraffic/YYYY/MM/DD/HH/
#      and concatenate them into a single hourly JSONL file.
#   3) Upload each hourly JSONL to two places:
#        a) History path: s3://<bucket>/monitoring/inference_jsonl/endpoint=<endpoint>/YYYY/MM/DD/HH/infer.jsonl
#        b) Latest path : s3://<bucket>/monitoring/inference_jsonl/latest/infer.jsonl  (overwritten every hour)
#
# Notes:
#   - Works with both .jsonl and .jsonl.gz capture files.
#   - We DO NOT transform the content; we just concatenate lines as-is. This keeps the format
#     fully compatible with SageMaker Model Monitor (BatchTransformInput).
#
# Usage (PowerShell):
#   python .\src\monitoring\capture_to_batchjsonl.py `
#       --endpoint bikeshare-staging `
#       --date 2025-09-22 `
#       --bucket mlops-bikeshare-387706002632-ca-central-1 `
#       --region ca-central-1 `
#       --out-prefix monitoring/inference_jsonl
#
#   # If your date is already formatted like 2025/09/22, that's fine too.
#
# Requirements:
#   pip install boto3 pandas (pandas not strictly required here, but often available already)

import argparse
import gzip
import io
from urllib.parse import urlparse

import boto3

# ---------- Utilities ----------


def normalize_date_str(s: str) -> str:
    """Accept 'YYYY-MM-DD' or 'YYYY/MM/DD' and return 'YYYY/MM/DD' (zero-padded)."""
    s = s.strip().replace("\\", "/")
    if "-" in s:
        y, m, d = s.split("-")
    else:
        y, m, d = s.split("/")
    return f"{int(y):04d}/{int(m):02d}/{int(d):02d}"


def s3_split_uri(s3_uri: str):
    """Split s3://bucket/prefix into (bucket, key_prefix) with no leading slash on key."""
    u = urlparse(s3_uri)
    if u.scheme != "s3":
        raise ValueError(f"Not an s3:// URI: {s3_uri}")
    return u.netloc, u.path.lstrip("/")


def s3_join(*parts: str) -> str:
    """Join S3 key parts with single slashes, ignoring empty segments."""
    cleaned = []
    for p in parts:
        if p is None:
            continue
        p = str(p).strip().strip("/")
        if p:
            cleaned.append(p)
    return "/".join(cleaned)


def iter_s3_keys(s3_client, bucket: str, prefix: str):
    """Yield object keys under a prefix using a paginator."""
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]


def read_capture_object(s3_client, bucket: str, key: str):
    """Read a capture object. Supports .jsonl and .jsonl.gz. Returns text (str)."""
    body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    if key.endswith(".gz"):
        # Decode gzip -> text
        with gzip.GzipFile(fileobj=io.BytesIO(body), mode="rb") as gz:
            return gz.read().decode("utf-8", errors="replace")
    else:
        # Direct UTF-8 decode
        return body.decode("utf-8", errors="replace")


def upload_text(s3_client, bucket: str, key: str, text: str):
    """Upload UTF-8 text to S3."""
    s3_client.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))
    return f"s3://{bucket}/{key}"


# ---------- Core logic ----------


def discover_capture_root(sm_client, s3_client, endpoint_name: str):
    """Return (bucket, capture_root_prefix) from endpoint-config's DataCaptureConfig.DestinationS3Uri.

    Example returned prefix (as seen in your project):
      datacapture/endpoint=bikeshare-staging-config-20250921-024033

    We will later append '/<endpoint-name>/AllTraffic/YYYY/MM/DD/HH/' to this root.
    """
    ep = sm_client.describe_endpoint(EndpointName=endpoint_name)
    cfg_name = ep["EndpointConfigName"]
    epcfg = sm_client.describe_endpoint_config(EndpointConfigName=cfg_name)
    dcc = epcfg.get("DataCaptureConfig", {})
    dest_uri = dcc.get("DestinationS3Uri")
    if not dest_uri:
        raise RuntimeError(f"Endpoint '{endpoint_name}' has no DataCapture DestinationS3Uri.")
    bucket, root_prefix = s3_split_uri(dest_uri)
    return bucket, root_prefix


def aggregate_hour(
    s3_client,
    capture_bucket: str,
    capture_root_prefix: str,
    endpoint_name: str,
    y: str,
    m: str,
    d: str,
    hh: str,
) -> str:
    """Concatenate all capture files under a given hour into one JSONL string.
    Returns the JSONL text (may be empty if no files found)."""
    # Capture hour prefix looks like:
    #   <root>/bikeshare-staging/AllTraffic/YYYY/MM/DD/HH/
    hour_prefix = s3_join(capture_root_prefix, endpoint_name, "AllTraffic", y, m, d, hh)
    keys = list(iter_s3_keys(s3_client, capture_bucket, hour_prefix))
    # Keep only .jsonl or .jsonl.gz
    keys = [k for k in keys if k.endswith(".jsonl") or k.endswith(".jsonl.gz")]

    if not keys:
        return ""

    parts = []
    for k in keys:
        txt = read_capture_object(s3_client, capture_bucket, k)
        # Ensure trailing newline so concatenation stays valid JSONL
        if txt and not txt.endswith("\n"):
            txt += "\n"
        parts.append(txt)
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True, help="SageMaker endpoint name (e.g., bikeshare-staging)")
    parser.add_argument("--date", required=True, help="UTC date: YYYY-MM-DD or YYYY/MM/DD")
    parser.add_argument("--bucket", required=True, help="Target S3 bucket for output JSONL (monitoring/... lives here)")
    parser.add_argument("--region", default="ca-central-1", help="AWS region of the endpoint")
    parser.add_argument("--out-prefix", default="monitoring/inference_jsonl", help="S3 prefix for output JSONL")
    parser.add_argument("--hours", default="all", help="Either 'all' or a single hour like '07'")
    args = parser.parse_args()

    # Parse date into Y/M/D (strings)
    ymd = normalize_date_str(args.date)  # 'YYYY/MM/DD'
    y, m, d = ymd.split("/")

    sm = boto3.client("sagemaker", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)

    # 1) Discover the capture root from endpoint-config
    cap_bucket, cap_root = discover_capture_root(sm, s3, args.endpoint)
    print(f"[info] Capture root: s3://{cap_bucket}/{cap_root}")

    # 2) Decide hours to process
    if args.hours.lower() == "all":
        hours = [f"{h:02d}" for h in range(24)]
    else:
        hh = int(args.hours)
        hours = [f"{hh:02d}"]

    # 3) For each hour, read → concatenate → upload to history and latest
    for hh in hours:
        print(f"[info] Aggregating hour {y}-{m}-{d} {hh}:00Z ...")
        jsonl_text = aggregate_hour(
            s3_client=s3,
            capture_bucket=cap_bucket,
            capture_root_prefix=cap_root,
            endpoint_name=args.endpoint,
            y=y,
            m=m,
            d=d,
            hh=hh,
        )

        if not jsonl_text:
            print(f"[warn] No capture files found for {y}/{m}/{d}/{hh}/ . Skipped.")
            continue

        # History key (kept for auditing)
        hist_key = s3_join(args.out_prefix, f"endpoint={args.endpoint}", y, m, d, hh, "infer.jsonl")
        hist_uri = upload_text(s3, args.bucket, hist_key, jsonl_text)
        print(f"[ok] Uploaded history file: {hist_uri}")

        # Latest key (overwritten each hour; used by Model Monitor schedule)
        latest_key = s3_join(args.out_prefix, "latest", "infer.jsonl")
        latest_uri = upload_text(s3, args.bucket, latest_key, jsonl_text)
        print(f"[ok] Uploaded latest file:  {latest_uri}")

    print("[done] Hourly aggregation to JSONL completed.")


if __name__ == "__main__":
    main()
