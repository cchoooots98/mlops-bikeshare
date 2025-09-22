# merge_labels_hour_to_latest.py
# Purpose: Merge all JSONL files under monitoring/quality/city=<city>/YYYY/MM/DD/HH/ into
#          s3://<bucket>/monitoring/quality/latest/labels.jsonl
# Assumes each line is {"inferenceId": "...", "groundTruthLabel": 0/1}

import argparse

import boto3


def list_keys(s3, bucket, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--city", default="nyc")
    ap.add_argument("--ymdh", required=True, help="UTC hour: YYYY/MM/DD/HH")
    ap.add_argument("--region", default="ca-central-1")
    args = ap.parse_args()

    s3 = boto3.client("s3", region_name=args.region)
    src_prefix = f"monitoring/quality/city={args.city}/{args.ymdh.strip('/')}/"
    keys = [k for k in list_keys(s3, args.bucket, src_prefix) if k.endswith(".jsonl") or k.endswith(".jsonl.gz")]
    if not keys:
        print(f"[warn] no JSONL found under s3://{args.bucket}/{src_prefix}")
        return

    # Concatenate all lines
    out_lines = []
    for k in keys:
        body = s3.get_object(Bucket=args.bucket, Key=k)["Body"].read()
        if k.endswith(".gz"):
            import gzip
            import io

            with gzip.GzipFile(fileobj=io.BytesIO(body), mode="rb") as gz:
                text = gz.read().decode("utf-8", errors="replace")
        else:
            text = body.decode("utf-8", errors="replace")
        if text and not text.endswith("\n"):
            text += "\n"
        out_lines.append(text)

    merged = "".join(out_lines)
    dst_key = "monitoring/quality/latest/labels.jsonl"
    s3.put_object(Bucket=args.bucket, Key=dst_key, Body=merged.encode("utf-8"), ContentType="application/json")
    print("[ok] wrote: s3://%s/%s" % (args.bucket, dst_key))


if __name__ == "__main__":
    main()
