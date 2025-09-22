# src/monitoring/capture_to_batchjsonl.py
# Purpose: Read endpoint data-capture files for a given UTC hour and write a single JSONL file
#          that contains flat records like: {"inferenceId":"...", "probability":0.62}
#          This normalized JSONL works with Model Monitor BatchTransformInput when you set:
#          - DatasetFormat = {"Json": {"Line": True}}
#          - ProbabilityAttribute = "probability"
#          - ProbabilityThresholdAttribute = "0.5"
import argparse
import gzip
import json

import boto3


def list_keys(s3, bucket, prefix):
    p = s3.get_paginator("list_objects_v2")
    for page in p.paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            k = o["Key"]
            if k.endswith(".jsonl") or k.endswith(".jsonl.gz"):
                yield k


def read_text(s3, bucket, key):
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    if key.endswith(".gz"):
        return gzip.decompress(body).decode("utf-8", errors="replace")
    return body.decode("utf-8", errors="replace")


def normalize_capture_line(line):
    """
    Convert one capture JSON line to {"inferenceId": "...", "probability": float}
    Assumes SageMaker capture schema and that endpoint output has {"predictions":[p]} in its 'data' string.
    """
    try:
        rec = json.loads(line)
        meta = rec.get("eventMetadata", {}) or {}
        inf_id = meta.get("inferenceId") or meta.get("eventId")  # fallback
        out_blob = (((rec.get("captureData") or {}).get("endpointOutput") or {}).get("data") or "").strip()
        # endpointOutput.data is itself a JSON string like: {"predictions":[0.6267]}\n
        prob = None
        if out_blob:
            try:
                out_json = json.loads(out_blob)
                preds = out_json.get("predictions")
                if isinstance(preds, list) and preds:
                    if isinstance(preds[0], (int, float)):
                        prob = float(preds[0])
                    elif isinstance(preds[0], list) and preds[0]:
                        prob = float(preds[0][0])
                    elif isinstance(preds[0], dict) and "score" in preds[0]:
                        prob = float(preds[0]["score"])
            except Exception:
                pass
        if inf_id is not None and prob is not None:
            return {"inferenceId": str(inf_id), "probability": float(prob)}
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--capture_root", required=True, help="e.g. datacapture/endpoint=bikeshare-staging-config-...")
    ap.add_argument("--ymdh", required=True, help="UTC hour: YYYY/MM/DD/HH")
    ap.add_argument("--out_prefix", default="monitoring/inference_jsonl/latest")
    ap.add_argument("--region", default="ca-central-1")
    args = ap.parse_args()

    s3 = boto3.client("s3", region_name=args.region)
    hour_prefix = f"{args.capture_root.strip('/')}/{args.endpoint}/AllTraffic/{args.ymdh.strip('/')}/"

    out_lines = []
    for key in list_keys(s3, args.bucket, hour_prefix):
        text = read_text(s3, args.bucket, key)
        for ln in text.splitlines():
            if not ln.strip():
                continue
            norm = normalize_capture_line(ln)
            if norm:
                out_lines.append(json.dumps(norm))

    if not out_lines:
        print(f"[warn] No usable capture records under s3://{args.bucket}/{hour_prefix}")
        return

    # Write to latest/ (overwrites) and to history/
    latest_key = f"{args.out_prefix.strip('/')}/infer.jsonl"
    s3.put_object(
        Bucket=args.bucket,
        Key=latest_key,
        Body=("\n".join(out_lines) + "\n").encode("utf-8"),
        ContentType="application/json",
    )
    hist_key = f"monitoring/inference_jsonl/endpoint={args.endpoint}/{args.ymdh.strip('/')}/infer.jsonl"
    s3.put_object(
        Bucket=args.bucket,
        Key=hist_key,
        Body=("\n".join(out_lines) + "\n").encode("utf-8"),
        ContentType="application/json",
    )
    print("[ok] wrote:", f"s3://{args.bucket}/{latest_key}", "and", f"s3://{args.bucket}/{hist_key}")


if __name__ == "__main__":
    main()
