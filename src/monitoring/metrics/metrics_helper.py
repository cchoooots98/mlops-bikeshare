"""
CloudWatch metrics helper for target-aware production monitoring.

All formal metrics use the same dimensions:
- Environment
- EndpointName
- City
- TargetName
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError


AWS_REGION = os.getenv("AWS_REGION", "eu-west-3")
CW_NAMESPACE = os.getenv("CW_NS", "Bikeshare/Model")
DEFAULT_ENDPOINT = os.getenv("SM_ENDPOINT", "bikeshare-bikes-prod")
DEFAULT_CITY = os.getenv("CITY", "paris")
DEFAULT_TARGET = os.getenv("TARGET_NAME", "bikes")
DEFAULT_ENVIRONMENT = os.getenv("SERVING_ENVIRONMENT", "production")

_CW = boto3.client("cloudwatch", region_name=AWS_REGION)


def build_metric_dimensions(
    *,
    endpoint: Optional[str] = None,
    city: Optional[str] = None,
    target_name: Optional[str] = None,
    environment: Optional[str] = None,
) -> List[Dict[str, str]]:
    return [
        {"Name": "Environment", "Value": environment or DEFAULT_ENVIRONMENT},
        {"Name": "EndpointName", "Value": endpoint or DEFAULT_ENDPOINT},
        {"Name": "City", "Value": city or DEFAULT_CITY},
        {"Name": "TargetName", "Value": target_name or DEFAULT_TARGET},
    ]


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _put_with_retry(metric_data: List[Dict], namespace: str = CW_NAMESPACE, max_attempts: int = 3) -> None:
    attempt = 0
    while True:
        try:
            _CW.put_metric_data(Namespace=namespace, MetricData=metric_data)
            return
        except ClientError:
            attempt += 1
            if attempt >= max_attempts:
                raise
            time.sleep(1.5 * attempt)


def put_metric(
    name: str,
    value: float,
    unit: str = "None",
    *,
    endpoint: Optional[str] = None,
    city: Optional[str] = None,
    target_name: Optional[str] = None,
    environment: Optional[str] = None,
    timestamp=None,
) -> None:
    if not _is_finite_number(value):
        return

    metric = {
        "MetricName": name,
        "Value": float(value),
        "Unit": unit,
        "Dimensions": build_metric_dimensions(
            endpoint=endpoint,
            city=city,
            target_name=target_name,
            environment=environment,
        ),
    }
    if timestamp is not None:
        metric["Timestamp"] = timestamp
    _put_with_retry([metric])


def put_metrics_bulk(
    items: Iterable[Tuple[str, float, str]],
    *,
    endpoint: Optional[str] = None,
    city: Optional[str] = None,
    target_name: Optional[str] = None,
    environment: Optional[str] = None,
    timestamp=None,
) -> None:
    dimensions = build_metric_dimensions(
        endpoint=endpoint,
        city=city,
        target_name=target_name,
        environment=environment,
    )
    metric_data: List[Dict] = []
    for name, value, unit in items:
        if not _is_finite_number(value):
            continue
        item = {
            "MetricName": name,
            "Value": float(value),
            "Unit": unit,
            "Dimensions": dimensions,
        }
        if timestamp is not None:
            item["Timestamp"] = timestamp
        metric_data.append(item)
    if metric_data:
        _put_with_retry(metric_data)


def publish_heartbeat(
    *,
    endpoint: Optional[str] = None,
    city: Optional[str] = None,
    target_name: Optional[str] = None,
    environment: Optional[str] = None,
) -> None:
    put_metric(
        "PredictionHeartbeat",
        1,
        "Count",
        endpoint=endpoint,
        city=city,
        target_name=target_name,
        environment=environment,
    )
