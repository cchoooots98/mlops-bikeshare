# lambda_publish_psi.py
# Purpose:
#   1) Read SageMaker Model Monitor outputs from S3: the latest "statistics.json"
#      (online distribution) under MONITOR_OUTPUT_PREFIX/statistics/
#   2) Read the training baseline "statistics.json" from BASELINE_STATISTICS_URI
#   3) Compute PSI per feature (numeric via histogram, categorical via top_values)
#   4) Aggregate PSI across features (max by default to be conservative)
#   5) Publish a single custom metric "PSI" to CloudWatch with dimensions:
#         EndpointName, City
#
# Cost notes:
#   - We publish ONE datapoint per run (hourly) -> cheapest custom metrics usage.
#   - We do not publish per-feature metrics by default (can be enabled if needed).
#
# Assumptions:
#   - The statistics.json structure is similar to Model Monitor outputs:
#       features: [
#         {
#           "name": "<feature_name>",
#           "numerical_statistics": {
#               "histogram": {
#                   "bin_edges": [...],       # length = k
#                   "bin_counts": [...],      # length = k-1
#               },
#               ...
#           }
#           or
#           "categorical_statistics": {
#               "top_values": [
#                   {"value": "A", "count": 123},
#                   {"value": "B", "count": 45},
#                   ...
#               ]
#           }
#         },
#         ...
#       ]
#   - If histograms exist but bin edges differ, we skip that feature (to avoid
#     rebin complexity). You can enhance rebinning if needed.
#
# Environment variables (must be set on the Lambda):
#   MONITOR_OUTPUT_PREFIX    -> e.g. s3://.../monitoring/data-quality/
#   BASELINE_STATISTICS_URI  -> e.g. s3://.../monitoring/data-quality/baseline/statistics.json
#   ENDPOINT_NAME            -> e.g. bikeshare-prod
#   CITY                     -> e.g. nyc
#   AGGREGATOR               -> "max" (default) or "mean"
#   LOG_TOPK                 -> optional, top-K features by PSI to log (default 5)
#
# IAM required:
#   s3:GetObject, s3:ListBucket for the relevant bucket/prefix
#   cloudwatch:PutMetricData
#   logs:CreateLogGroup/Stream, logs:PutLogEvents (via AWSLambdaBasicExecutionRole)
#
import json
import logging
import math
import os
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")
cw = boto3.client("cloudwatch", region_name=os.getenv("AWS_REGION", "ca-central-1"))


def _parse_s3_uri(uri: str):
    # Comment: Parse s3://bucket/prefix/... into (bucket, key)
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an s3 URI: {uri}")
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def _load_json_s3(s3_uri: str) -> dict:
    # Comment: Download and parse a JSON file from S3.
    b, k = _parse_s3_uri(s3_uri)
    obj = s3.get_object(Bucket=b, Key=k)
    return json.loads(obj["Body"].read())


def _find_latest_statistics_uri(prefix_uri: str) -> str:
    # Comment: Support both "<prefix>/statistics/YYYY/MM/DD/HH/statistics.json"
    # and "<prefix>/YYYY/MM/DD/HH/statistics.json" layouts.
    bucket, prefix = _parse_s3_uri(prefix_uri)
    candidates = [
        prefix.rstrip("/") + "/statistics/",  # legacy layout
        prefix.rstrip("/") + "/",  # model monitor default layout
    ]

    paginator = s3.get_paginator("list_objects_v2")
    latest_key, latest_mtime = None, None

    for base in candidates:
        for page in paginator.paginate(Bucket=bucket, Prefix=base):
            for item in page.get("Contents", []):
                key = item["Key"]
                if key.endswith("statistics.json"):
                    mtime = item["LastModified"]
                    if latest_mtime is None or mtime > latest_mtime:
                        latest_mtime, latest_key = mtime, key

    if not latest_key:
        # Comment: Be explicit about where we looked for easier debugging.
        raise FileNotFoundError(
            f"No statistics.json found under s3://{bucket}/{candidates[0]} or s3://{bucket}/{candidates[1]}"
        )
    return f"s3://{bucket}/{latest_key}"


def _histogram_to_proportions(h: dict) -> list:
    # Comment: Convert histogram counts to proportions.
    # Expected keys: bin_edges (len k), bin_counts (len k-1).
    counts = h.get("bin_counts") or h.get("counts") or []
    total = float(sum(counts)) if counts else 0.0
    if total <= 0.0:
        return [0.0 for _ in counts]
    return [c / total for c in counts]


def _extract_numeric_hist(feature: dict):
    # Comment: Try direct histogram first: {bin_edges, bin_counts}
    num = feature.get("numerical_statistics") or {}
    hist = num.get("histogram") or {}
    edges = hist.get("bin_edges")
    counts = hist.get("bin_counts") or hist.get("counts")
    if edges and counts and len(edges) == len(counts) + 1:
        return {"bin_edges": edges, "bin_counts": counts}

    # Comment: Fall back to KLL buckets: a list of {lower_bound, upper_bound, count}
    kll = (num.get("distribution") or {}).get("kll") or {}
    buckets = kll.get("buckets")
    if buckets:
        # Comment: Build edges as [lower0, upper0, upper1, ...], counts from each bucket
        edges_out, counts_out = [buckets[0]["lower_bound"]], []
        for b in buckets:
            edges_out.append(b.get("upper_bound", b.get("lower_bound")))
            counts_out.append(float(b.get("count", 0)))
        return {"bin_edges": edges_out, "bin_counts": counts_out}

    # Comment: No usable numeric summary found
    return None


def _extract_categorical_dist(feature: dict):
    # Comment: Extract a dict {category_value -> proportion} from top_values if present.
    cat = feature.get("categorical_statistics") or {}
    topv = cat.get("top_values")
    if not topv:
        return None
    total = float(sum([tv.get("count", 0) for tv in topv]))
    if total <= 0.0:
        return {}
    return {str(tv.get("value")): tv.get("count", 0) / total for tv in topv}


def _psi(p: float, q: float, eps: float = 1e-6) -> float:
    # Comment: PSI component for a single bin/category, numerically stable with epsilon.
    p = max(p, eps)
    q = max(q, eps)
    return (p - q) * math.log(p / q)


def _psi_from_hist(baseline_hist: dict, latest_hist: dict) -> float:
    # Comment: Compute PSI from two histograms with the SAME bin_edges length and values.
    be = baseline_hist["bin_edges"]
    le = latest_hist["bin_edges"]
    if len(be) != len(le) or any(abs(b - le_edge) > 1e-12 for b, le_edge in zip(be, le)):
        # Comment: Edges differ -> skip to avoid rebin complexity (keep it simple & safe).
        raise ValueError("Bin edges differ; rebinning not implemented.")
    p = _histogram_to_proportions(baseline_hist)
    q = _histogram_to_proportions(latest_hist)
    return sum(_psi(pi, qi) for pi, qi in zip(p, q))


def _psi_from_categorical(baseline_dist: dict, latest_dist: dict) -> float:
    # Comment: Build union of categories, fill missing with 0, then sum PSI.
    keys = set(baseline_dist.keys()) | set(latest_dist.keys())
    total = 0.0
    for k in keys:
        p = baseline_dist.get(k, 0.0)
        q = latest_dist.get(k, 0.0)
        total += _psi(p, q)
    return total


def _index_features(stat_json: dict) -> dict:
    # Comment: Build a dict {feature_name -> feature_entry}
    feats = stat_json.get("features") or stat_json.get("feature_metrics") or []
    out = {}
    for f in feats:
        name = f.get("name") or f.get("feature_name")
        if name:
            out[name] = f
    return out


def handler(event, context):
    # ----------- Read env -----------
    monitor_prefix = os.getenv("MONITOR_OUTPUT_PREFIX")  # s3://.../monitoring/data-quality/
    baseline_uri = os.getenv("BASELINE_STATISTICS_URI")  # s3://.../baseline/statistics.json
    endpoint_name = os.getenv("ENDPOINT_NAME", "bikeshare-prod")
    city = os.getenv("CITY", "nyc")
    aggregator = os.getenv("AGGREGATOR", "max").lower()  # "max" or "mean"
    log_topk = int(os.getenv("LOG_TOPK", "5"))

    if not monitor_prefix or not baseline_uri:
        raise RuntimeError("MONITOR_OUTPUT_PREFIX and BASELINE_STATISTICS_URI must be set.")

    # ----------- Locate & load JSONs -----------
    latest_stats_uri = _find_latest_statistics_uri(monitor_prefix)
    logger.info(f"Latest statistics: {latest_stats_uri}")
    baseline = _load_json_s3(baseline_uri)
    latest = _load_json_s3(latest_stats_uri)

    base_idx = _index_features(baseline)
    live_idx = _index_features(latest)

    # ----------- Compute PSI per feature -----------
    feature_psis = []
    skipped = []

    for feat_name, base_feat in base_idx.items():
        live_feat = live_idx.get(feat_name)
        if not live_feat:
            skipped.append((feat_name, "missing_in_latest"))
            continue

        # Try numeric histogram first
        base_hist = _extract_numeric_hist(base_feat)
        live_hist = _extract_numeric_hist(live_feat)

        try:
            if base_hist and live_hist:
                val = _psi_from_hist(base_hist, live_hist)
                feature_psis.append((feat_name, val))
                continue
        except Exception as e:
            skipped.append((feat_name, f"hist_failed:{e}"))

        # Try categorical top_values
        base_cat = _extract_categorical_dist(base_feat)
        live_cat = _extract_categorical_dist(live_feat)
        if base_cat is not None and live_cat is not None:
            val = _psi_from_categorical(base_cat, live_cat)
            feature_psis.append((feat_name, val))
            continue

        skipped.append((feat_name, "no_supported_stats"))

    if not feature_psis:
        # If we cannot compute anything, publish zero and log a warning
        logger.warning("No feature PSI computed. Check histogram/top_values availability.")
        final_psi = 0.0
    else:
        # Aggregate
        vals = [v for _, v in feature_psis]
        final_psi = max(vals) if aggregator == "max" else (sum(vals) / len(vals))

    # ----------- Log top-k details for debugging -----------
    feature_psis.sort(key=lambda x: x[1], reverse=True)
    logger.info("Top feature drifts by PSI:")
    for name, v in feature_psis[:log_topk]:
        logger.info(f"  {name}: {v:.4f}")
    if skipped:
        logger.info(f"Skipped features: {len(skipped)} (see reasons in logs)")

    # ----------- Publish CloudWatch metric -----------
    try:
        cw.put_metric_data(
            Namespace="Bikeshare/Model",
            MetricData=[
                {
                    "MetricName": "PSI",
                    "Value": float(final_si := final_psi),
                    "Unit": "None",
                    "Dimensions": [
                        {"Name": "EndpointName", "Value": endpoint_name},
                        {"Name": "City", "Value": city},
                    ],
                }
            ],
        )
        logger.info(f"Published PSI={final_si:.4f} for EndpointName={endpoint_name}, City={city}")
    except ClientError as e:
        logger.error(f"Failed to publish metric: {e}")
        raise

    return {"psi": final_psi, "features_computed": len(feature_psis), "features_skipped": len(skipped)}
