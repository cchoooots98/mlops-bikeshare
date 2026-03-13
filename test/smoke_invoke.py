#!/usr/bin/env python3
import argparse
import json
import os
import sys

import boto3

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.features.schema import FEATURE_COLUMNS
from src.model_package import load_package_manifest, resolve_active_package_dir


def resolve_feature_columns(
    model_package_dir: str | None = None,
    deployment_state_path: str | None = None,
) -> list[str]:
    if not model_package_dir and not deployment_state_path:
        return FEATURE_COLUMNS
    package_dir = resolve_active_package_dir(
        model_package_dir=model_package_dir,
        deployment_state_path=deployment_state_path,
    )
    return list(load_package_manifest(package_dir)["feature_columns"])


def sample_feature_row(feature_columns: list[str] | None = None) -> list[float]:
    columns = feature_columns or FEATURE_COLUMNS
    sample = {
        "minutes_since_prev_snapshot": 5.0,
        "util_bikes": 0.60,
        "util_docks": 0.40,
        "delta_bikes_5m": -1.0,
        "delta_docks_5m": 1.0,
        "roll15_net_bikes": -2.0,
        "roll30_net_bikes": -3.0,
        "roll60_net_bikes": -4.0,
        "roll15_bikes_mean": 12.0,
        "roll30_bikes_mean": 11.5,
        "roll60_bikes_mean": 10.9,
        "nbr_bikes_weighted": 7.2,
        "nbr_docks_weighted": 8.8,
        "has_neighbors_within_radius": 1.0,
        "neighbor_count_within_radius": 4.0,
        "hour": 8.0,
        "dow": 2.0,
        "is_weekend": 0.0,
        "is_holiday": 0.0,
        "temperature_c": 20.5,
        "humidity_pct": 60.0,
        "wind_speed_ms": 4.3,
        "precipitation_mm": 0.0,
        "weather_code": 2.0,
        "hourly_temperature_c": 20.0,
        "hourly_humidity_pct": 58.0,
        "hourly_wind_speed_ms": 4.8,
        "hourly_precipitation_mm": 0.0,
        "hourly_precipitation_probability_pct": 15.0,
        "hourly_weather_code": 2.0,
    }
    return [sample[column] for column in columns]


def build_payload(feature_columns: list[str] | None = None) -> dict:
    columns = feature_columns or FEATURE_COLUMNS
    return {"inputs": {"dataframe_split": {"columns": columns, "data": [sample_feature_row(columns)]}}}


def main():
    parser = argparse.ArgumentParser(description="Invoke SageMaker endpoint.")
    parser.add_argument("--endpoint-name", required=True, help="Endpoint name.")
    parser.add_argument("--region", required=True, help="AWS region, e.g., eu-west-3.")
    parser.add_argument("--model-package-dir", default=None, help="Optional local model package directory.")
    parser.add_argument("--deployment-state-path", default=None, help="Optional deployment state JSON path.")
    args = parser.parse_args()

    smrt = boto3.client("sagemaker-runtime", region_name=args.region)
    feature_columns = resolve_feature_columns(
        model_package_dir=args.model_package_dir,
        deployment_state_path=args.deployment_state_path,
    )
    resp = smrt.invoke_endpoint(
        EndpointName=args.endpoint_name,
        ContentType="application/json",
        Body=json.dumps(build_payload(feature_columns)).encode("utf-8"),
    )
    print(resp["Body"].read().decode("utf-8"))


if __name__ == "__main__":
    main()
