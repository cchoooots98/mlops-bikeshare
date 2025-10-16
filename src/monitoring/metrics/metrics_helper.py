"""
Lightweight CloudWatch metrics helper used across the project.

Goals:
- Single place to publish custom metrics with consistent dimensions
  (EndpointName, City) so the Streamlit dashboard and alarms "just work".
- Small, dependency-free (only boto3) and safe on Windows runners and Lambda.
- Retry on throttling and skip None/NaN values gracefully.

Usage:
    from src.monitoring.metrics.metrics_helper import (
        put_metric, put_metrics_bulk, publish_heartbeat
    )

    put_metric("PR-AUC-24h", 0.9582)           # Unit defaults to "None"
    put_metric("Samples-24h", 12345, "Count")
    publish_heartbeat()                        # One count per successful batch
"""

from __future__ import annotations
import math
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

# ---- Centralized configuration (keeps staging/prod flips simple) ----
AWS_REGION = os.getenv("AWS_REGION", "ca-central-1")
CW_NAMESPACE = os.getenv("CW_NS", "Bikeshare/Model")
DEFAULT_ENDPOINT = os.getenv("SM_ENDPOINT", "bikeshare-prod")
DEFAULT_CITY = os.getenv("CITY", "nyc")

# Create one global client (boto3 is thread-safe for simple use cases).
_CW = boto3.client("cloudwatch", region_name=AWS_REGION)


def _dims(endpoint: Optional[str] = None, city: Optional[str] = None) -> List[Dict[str, str]]:
    """Build the required CloudWatch dimensions for our dashboards/alarms."""
    return [
        {"Name": "EndpointName", "Value": endpoint or DEFAULT_ENDPOINT},
        {"Name": "City", "Value": city or DEFAULT_CITY},
    ]


def _is_finite_number(x: object) -> bool:
    """Return True if x is a finite float/int (skip None/NaN/inf)."""
    try:
        xf = float(x)
        return math.isfinite(xf)
    except Exception:
        return False


def _put_with_retry(metric_data: List[Dict], namespace: str = CW_NAMESPACE, max_attempts: int = 3) -> None:
    """Publish metric data with a tiny retry loop for throttling."""
    attempt = 0
    while True:
        try:
            _CW.put_metric_data(Namespace=namespace, MetricData=metric_data)
            return
        except ClientError as e:
            attempt += 1
            if attempt >= max_attempts:
                raise
            # Exponential backoff on throttling or transient errors.
            time.sleep(1.5 * attempt)


def put_metric(
    name: str,
    value: float,
    unit: str = "None",
    *,
    endpoint: Optional[str] = None,
    city: Optional[str] = None,
    timestamp=None,
) -> None:
    """
    Publish a single scalar metric with our standard dimensions.

    - Skips None/NaN/inf silently (so callers don't need to guard every call).
    - Unit can be "None", "Count", "Milliseconds", etc.
    """
    if not _is_finite_number(value):
        return

    md = [{
        "MetricName": name,
        "Value": float(value),
        "Unit": unit,
        "Dimensions": _dims(endpoint, city),
    }]
    if timestamp is not None:
        md[0]["Timestamp"] = timestamp
    _put_with_retry(md)


def put_metrics_bulk(
    items: Iterable[Tuple[str, float, str]],
    *,
    endpoint: Optional[str] = None,
    city: Optional[str] = None,
    timestamp=None,
) -> None:
    """
    Publish multiple (name, value, unit) tuples in one CloudWatch call.

    Example:
        put_metrics_bulk([
            ("PR-AUC-24h", 0.9582, "None"),
            ("F1-24h", 0.856, "None"),
            ("Samples-24h", 12345, "Count"),
        ])
    """
    md: List[Dict] = []
    dims = _dims(endpoint, city)
    for (name, value, unit) in items:
        if not _is_finite_number(value):
            continue
        item = {
            "MetricName": name,
            "Value": float(value),
            "Unit": unit,
            "Dimensions": dims,
        }
        if timestamp is not None:
            item["Timestamp"] = timestamp
        md.append(item)

    if md:
        _put_with_retry(md)


def publish_heartbeat(*, endpoint: Optional[str] = None, city: Optional[str] = None) -> None:
    """
    Publish one "PredictionHeartbeat" Count=1.
    Call this after each successful prediction batch (or once per 10–15 min).
    """
    put_metric("PredictionHeartbeat", 1, "Count", endpoint=endpoint, city=city)

