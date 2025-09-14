#!/usr/bin/env python3
# ruff: noqa: E402
# -*- coding: utf-8 -*-
"""
Smoke invoke for SageMaker endpoint.

- Keeps all imports at the top (fixes Ruff E402).
- Uses FEATURE_COLUMNS from your project to ensure the request schema matches training.
- Sends a minimal one-row payload to the endpoint and prints the response.
"""
import argparse
import json
import os
import sys

import boto3
import numpy as np

# Import the same schema used in training so names/order match exactly.
# --- Make project root importable first (so 'schema.py' in repo wins over PyPI 'schema') ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../test
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))  # repo root
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# --- Import the training feature list from the project schema ---
from src.features.schema import FEATURE_COLUMNS  # defined in your project's schema.py (25 features)


def main():
    parser = argparse.ArgumentParser(description="Invoke SageMaker endpoint.")
    parser.add_argument("--endpoint-name", required=True, help="Endpoint name.")
    parser.add_argument("--region", required=True, help="AWS region, e.g., ca-central-1.")
    args = parser.parse_args()

    smrt = boto3.client("sagemaker-runtime", region_name=args.region)

    row = np.array([
            0.60,
            0.40,  # util_bikes, util_docks (0..1)
            -1.0,
            1.0,  # delta_bikes_5m, delta_docks_5m
            -2.0,
            -3.0,
            -4.0,  # roll15_net_bikes, roll30_net_bikes, roll60_net_bikes
            12.0,
            11.5,
            10.9,  # roll15_bikes_mean, roll30_bikes_mean, roll60_bikes_mean
            7.2,
            8.8,  # nbr_bikes_weighted, nbr_docks_weighted
            8,
            2,
            0,
            0,  # hour, dow, is_weekend, is_holiday
            20.5,
            0.0,
            10.0,  # temp_c, precip_mm, wind_kph
            60.0,
            1012.0,  # rhum_pct, pres_hpa
            180.0,
            15.0,  # wind_dir_deg, wind_gust_kph
            0.0,
            2,  # snow_mm, weather_code
        ], dtype=np.float32).tolist()  # <-- ensure float32

    payload = {"inputs": {"dataframe_split": {"columns": FEATURE_COLUMNS, "data": [row]}}}

    resp = smrt.invoke_endpoint(
        EndpointName=args.endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    print(resp["Body"].read().decode("utf-8"))


if __name__ == "__main__":
    main()
