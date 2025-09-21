# backfill_mq_jsonl_from_ds_parquet_v2.py
# Purpose: Convert joined Parquet under city=<city>/ds=<YYYY-MM-DD>/ into Ground Truth JSONL
#          under city=<city>/<YYYY>/<MM>/<DD>/<HH>/labels-<YYYY-MM-DD-HH>.jsonl
# JSONL schema required by AWS:
# {
#   "groundTruthData": {"data": "1", "encoding": "CSV"},
#   "eventMetadata": {"inferenceId": "<dt>_<station_id>"},
#   "eventVersion": "0"
# }

import json
import sys

import boto3
import pandas as pd


def write_jsonl_s3(lines, bucket, key):
    """Write lines (list of JSON strings) as JSONL to S3 with application/json content-type."""
    body = ("\n".join(lines) + "\n").encode("utf-8")
    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")


def run(bucket, city, ds):
    src = f"s3://{bucket}/monitoring/quality/city={city}/ds={ds}/"
    df = pd.read_parquet(src)
    if df.empty:
        print(f"[backfill-v2] no rows at {src}")
        return

    required = {"station_id", "dt", "y_stockout_bikes_30"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in Parquet: {missing}")

    # Extract hour parts from dt like "YYYY-MM-DD-HH-mm"
    df["HH"] = df["dt"].astype(str).str.slice(11, 13)
    df["YYYY"] = df["dt"].astype(str).str.slice(0, 4)
    df["MM"] = df["dt"].astype(str).str.slice(5, 7)
    df["DD"] = df["dt"].astype(str).str.slice(8, 10)

    for (yyyy, mm, dd, hh), g in df.groupby(["YYYY", "MM", "DD", "HH"]):
        lines = []
        for station_id, dt_str, label in g[["station_id", "dt", "y_stockout_bikes_30"]].itertuples(index=False):
            inference_id = f"{dt_str}_{station_id}"
            lines.append(
                json.dumps(
                    {
                        "groundTruthData": {"data": str(float(label)), "encoding": "CSV"},
                        "eventMetadata": {"inferenceId": inference_id},
                        "eventVersion": "0",
                    },
                    ensure_ascii=False,
                )
            )

        dst = f"monitoring/quality/city={city}/{yyyy}/{mm}/{dd}/{hh}/labels-{yyyy}-{mm}-{dd}-{hh}.jsonl"
        write_jsonl_s3(lines, bucket, dst)
        print(f"[backfill-v2] wrote {len(lines)} -> s3://{bucket}/{dst}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python src/monitoring/backfill_mq_jsonl_from_ds_parquet_v2.py <bucket> <city> <YYYY-MM-DD>")
        sys.exit(2)
    run(sys.argv[1], sys.argv[2], sys.argv[3])
