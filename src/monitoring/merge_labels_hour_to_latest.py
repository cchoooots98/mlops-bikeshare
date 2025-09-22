# src/monitoring/merge_labels_hour_to_latest.py
# Purpose: Merge all labels of one UTC hour into Model Monitor's required JSONL format.
# Input sources (pick what's available under the hour partition):
#   - monitoring/quality/city=<city>/<YYYY>/<MM>/<DD>/<HH>/*.jsonl   (with inferenceId + label)  OR
#   - monitoring/quality/city=<city>/ds=<YYYY-MM-DD>/part-*.parquet   (must contain inference_id and y_stockout_bikes_30)
# Output:
#   - monitoring/quality/latest/labels.jsonl   (overwrite)
#   - monitoring/quality/city=<city>/<YYYY>/<MM>/<DD>/<HH>/labels-...jsonl  (for auditing)
import argparse
import gzip
import io
import json

import boto3
import pandas as pd


def list_keys(s3, bucket, prefix):
    p = s3.get_paginator("list_objects_v2")
    for page in p.paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            yield o["Key"]


def read_text(s3, bucket, key):
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    if key.endswith(".gz"):
        return gzip.decompress(body).decode("utf-8", errors="replace")
    return body.decode("utf-8", errors="replace")


def from_jsonl_hour(s3, bucket, prefix):
    lines = []
    for k in list_keys(s3, bucket, prefix):
        if k.endswith(".jsonl") or k.endswith(".jsonl.gz"):
            txt = read_text(s3, bucket, k)
            for ln in txt.splitlines():
                if not ln.strip():
                    continue
                try:
                    obj = json.loads(ln)
                    inf = obj.get("inferenceId") or obj.get("inference_id")
                    lbl = obj.get("groundTruthLabel") or obj.get("label") or obj.get("y")
                    if inf is not None and lbl is not None:
                        lines.append({"inferenceId": str(inf), "label": int(lbl)})
                except Exception:
                    pass
    return lines


def from_parquet_day(s3, bucket, day_prefix, utc_hour):
    # Read all parquet parts for ds=<YYYY-MM-DD>, filter by hour from your 'dt' if present
    # Expect columns: inference_id, y_stockout_bikes_30
    keys = [k for k in list_keys(s3, bucket, day_prefix) if k.endswith(".parquet")]
    if not keys:
        return []
    dfs = []
    for k in keys:
        body = s3.get_object(Bucket=bucket, Key=k)["Body"].read()
        df = pd.read_parquet(io.BytesIO(body))
        dfs.append(df)
    if not dfs:
        return []
    df = pd.concat(dfs, ignore_index=True)
    # If you have local time 'dt', we skip filtering by exact hour; rely on correct partition passed to the script.
    if "inference_id" not in df.columns or "y_stockout_bikes_30" not in df.columns:
        return []
    out = []
    for inf, lbl in zip(df["inference_id"].astype(str), df["y_stockout_bikes_30"].astype(int)):
        out.append({"inferenceId": inf, "label": int(lbl)})
    return out


def to_mm_groundtruth_line(inf_id, label_int):
    # Wrap into Model Monitor groundTruthData structure (encoding CSV for a scalar label)
    return {
        "groundTruthData": {"data": str(int(label_int)), "encoding": "CSV"},
        "eventMetadata": {"inferenceId": str(inf_id)},
        "eventVersion": "0",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--city", default="nyc")
    ap.add_argument("--ymdh", required=True, help="UTC hour: YYYY/MM/DD/HH")
    ap.add_argument("--region", default="ca-central-1")
    args = ap.parse_args()

    s3 = boto3.client("s3", region_name=args.region)
    y, m, d, h = args.ymdh.split("/")
    src_prefix_jsonl = f"monitoring/quality/city={args.city}/{args.ymdh}/"
    src_prefix_parquet = f"monitoring/quality/city={args.city}/ds={y}-{m}-{d}/"

    pairs = from_jsonl_hour(s3, args.bucket, src_prefix_jsonl)
    if not pairs:
        pairs = from_parquet_day(s3, args.bucket, src_prefix_parquet, args.ymdh)
    if not pairs:
        print(f"[warn] No labels found under hour='{src_prefix_jsonl}' nor day='{src_prefix_parquet}'")
        return

    lines = [json.dumps(to_mm_groundtruth_line(p["inferenceId"], p["label"])) for p in pairs]
    text = "\n".join(lines) + "\n"

    # write latest
    latest_key = "monitoring/quality/latest/labels.jsonl"
    s3.put_object(Bucket=args.bucket, Key=latest_key, Body=text.encode("utf-8"), ContentType="application/json")
    # write hourly
    hourly_key = f"monitoring/quality/city={args.city}/{args.ymdh}/labels-{y}-{m}-{d}-{h}.jsonl"
    s3.put_object(Bucket=args.bucket, Key=hourly_key, Body=text.encode("utf-8"), ContentType="application/json")
    print("[ok] wrote:", f"s3://{args.bucket}/{latest_key}", "and", f"s3://{args.bucket}/{hourly_key}")


if __name__ == "__main__":
    main()
