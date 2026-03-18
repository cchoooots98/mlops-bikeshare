from __future__ import annotations

from datetime import datetime, timedelta, timezone

import boto3
import botocore.exceptions
import pandas as pd

from src.monitoring.metrics.metrics_helper import build_metric_dimensions


def create_cloudwatch_client(*, region_name: str, profile_name: str | None = None):
    session = boto3.Session(profile_name=profile_name) if profile_name else boto3.Session()
    return session.client("cloudwatch", region_name=region_name)


def build_dashboard_metric_dimensions(*, environment: str, endpoint_name: str, city: str, target_name: str) -> dict[str, str]:
    return {
        item["Name"]: item["Value"]
        for item in build_metric_dimensions(
            environment=environment,
            endpoint=endpoint_name,
            city=city,
            target_name=target_name,
        )
    }


def fetch_metric_series(
    cw,
    *,
    namespace: str,
    metric_name: str,
    dimensions: dict[str, str],
    minutes: int = 24 * 60,
    period: int = 300,
    stat: str = "Average",
) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes)
    query = [
        {
            "Id": "m1",
            "MetricStat": {
                "Metric": {
                    "Namespace": namespace,
                    "MetricName": metric_name,
                    "Dimensions": [{"Name": key, "Value": value} for key, value in dimensions.items()],
                },
                "Period": period,
                "Stat": stat,
            },
            "ReturnData": True,
        }
    ]
    try:
        response = cw.get_metric_data(
            StartTime=start,
            EndTime=end,
            MetricDataQueries=query,
            ScanBy="TimestampAscending",
        )
    except botocore.exceptions.ClientError:
        return pd.DataFrame({"ts": [], metric_name: []})
    result = response["MetricDataResults"][0]
    return pd.DataFrame({"ts": result.get("Timestamps", []), metric_name: result.get("Values", [])})
