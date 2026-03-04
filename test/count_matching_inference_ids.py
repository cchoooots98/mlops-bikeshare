# count_matching_inference_ids.py
import gzip
import io
import json

import boto3
import requests


def _open_stream_from_uri(uri: str):
    """
    Return a bytes stream (BytesIO) for s3://, https://, or local file paths.
    If https is forbidden/not found for an S3 website-style URL, fall back to boto3.
    """
    import re

    if uri.startswith("s3://"):
        from urllib.parse import urlparse

        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        return io.BytesIO(obj["Body"].read())

    if uri.startswith(("http://", "https://")):
        try:
            r = requests.get(uri, timeout=60)
            r.raise_for_status()
            return io.BytesIO(r.content)
        except requests.HTTPError as e:
            # Try to parse aws S3 host and fall back to boto3
            m = re.match(r"https?://([^/]+)\.s3(?:[.-]([a-z0-9-]+))?\.amazonaws\.com/(.+)", uri)
            if m and e.response is not None and e.response.status_code in (403, 404):
                bucket, _region, key = m.groups()
                s3 = boto3.client("s3")
                obj = s3.get_object(Bucket=bucket, Key=key)
                return io.BytesIO(obj["Body"].read())
            raise

    # local file
    with open(uri, "rb") as f:
        return io.BytesIO(f.read())


def _iter_jsonl(stream: io.BytesIO, name: str):
    """
    Yield dicts from a newline-delimited JSON file.
    Handles plain text or gzip (.gz) by sniffing filename or gzip header.
    """
    # Detect gzip: either name endswith .gz or header magic number
    pos = stream.tell()
    head = stream.read(2)
    stream.seek(pos)
    is_gz = name.endswith(".gz") or head == b"\x1f\x8b"

    text_stream = gzip.GzipFile(fileobj=stream, mode="rb") if is_gz else stream
    for i, raw in enumerate(text_stream):
        # raw is bytes; skip empty/whitespace lines
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"{name}: JSON decode error on line {i+1}: {e}\nLine: {line[:200]}...")


def _extract_inference_id(obj: dict):
    """
    Try to extract an identifier from eventMetadata.
    Prefer 'inferenceId'; fall back to 'eventId' if needed.
    """
    md = obj.get("eventMetadata", {}) or {}
    return md.get("inferenceId") or md.get("eventId")


def collect_ids(uri: str, label: str):
    stream = _open_stream_from_uri(uri)
    ids = []
    for rec in _iter_jsonl(stream, label):
        iid = _extract_inference_id(rec)
        if iid:
            ids.append(iid)
    return ids


def main():
    # === Replace with your two paths ===
    capture_uri = "s3://mlops-bikeshare-387706002632-ca-central-1/datacapture/endpoint=bikeshare-staging-config-20250923-043944/bikeshare-staging/AllTraffic/2025/09/23/05/02-45-884-3d05d4c6-29d2-4962-ad4c-353f51225ac6.jsonl"
    gt_uri = "https://mlops-bikeshare-387706002632-ca-central-1.s3.ca-central-1.amazonaws.com/monitoring/ground-truth/2025/09/23/04/labels-2025092304.jsonl"

    cap_ids = collect_ids(capture_uri, "capture")
    gt_ids = collect_ids(gt_uri, "ground-truth")

    cap_set = set(cap_ids)
    gt_set = set(gt_ids)
    inter = cap_set & gt_set

    print("==== Summary ====")
    print(f"Capture total lines with id:     {len(cap_ids)}")
    print(f"Ground-truth total lines with id:{len(gt_ids)}")
    print(f"Unique capture ids:              {len(cap_set)}")
    print(f"Unique ground-truth ids:         {len(gt_set)}")
    print(f"Matching ids (intersection):     {len(inter)}")

    # Optional: show a few examples for spot-check
    preview = list(inter)[:5]
    if preview:
        print("\nExamples of matching inferenceId/eventId:")
        for x in preview:
            print(f"  {x}")

    # Optional: quick diagnostics of non-overlap
    if len(inter) == 0:
        only_cap = list(cap_set - gt_set)[:3]
        only_gt = list(gt_set - cap_set)[:3]
        if only_cap:
            print("\nIDs only in capture (first 3):")
            for x in only_cap:
                print(f"  {x}")
        if only_gt:
            print("\nIDs only in ground-truth (first 3):")
            for x in only_gt:
                print(f"  {x}")


if __name__ == "__main__":
    main()
