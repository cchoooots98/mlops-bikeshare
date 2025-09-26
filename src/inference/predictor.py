# src/inference/predictor.py
# Windows/VSCode friendly. All comments are in English as requested.
# Purpose:
#   - Build ONLINE features ONLY for the latest 5-min snapshot (city-wide).
#   - Do BATCH inference against the SageMaker real-time endpoint (avoid row-wise HTTP).
#   - Write one parquet shard per dt:
#       s3://{bucket}/inference/city={city}/dt={latest_dt}/predictions.parquet
#
# Expected environment / config:
#   - Reuse read_env()/athena_conn() from your existing codebase (build_features.py).
#   - SM_ENDPOINT comes from environment variable (default "bikeshare-staging").
#
# Notes:
#   - This file is intentionally lean: no backfill, no actuals, no quality join.
#   - Keep the endpoint warm and the model small to hit ≤2 minutes end-to-end.

import io
import json
import os
import warnings
from typing import List

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Reuse your existing modules
from src.features.build_features import athena_conn, read_env
from src.features.schema import FEATURE_COLUMNS
from src.inference.featurize_online import build_online_features

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

# ---------- AWS clients ----------


def _s3():
    """Get a boto3 S3 client."""
    return boto3.client("s3")


def _smr():
    """Get a boto3 SageMaker runtime client."""
    return boto3.client("sagemaker-runtime")


# ---------- S3 helpers ----------


def _write_parquet_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    """Write a small dataframe to S3 as a single Parquet object using an in-memory buffer."""
    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    _s3().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


def _prediction_key(city: str, dt: str) -> str:
    """S3 key for per-dt predictions parquet."""
    return f"inference/city={city}/dt={dt}/predictions.parquet"


# ---------- Inference (BATCH) ----------


def _invoke_endpoint_batch(endpoint: str, batch_rows: List[list]) -> list:
    """
    Send a single batch to the real-time endpoint and return a list of predictions.
    This assumes the model follows the MLflow pyfunc/pandas "dataframe_split" schema
    and returns either {"predictions":[...]} or a raw list.
    """
    rt = _smr()
    payload = {
        "inputs": {
            "dataframe_split": {
                "columns": FEATURE_COLUMNS,
                "data": batch_rows,  # a list of float rows
            }
        }
    }
    resp = rt.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    body = resp["Body"].read().decode("utf-8", errors="ignore")
    out = json.loads(body) if body.strip().startswith("{") or body.strip().startswith("[") else body

    # Normalize common shapes
    preds = out.get("predictions", out) if isinstance(out, dict) else out
    if not isinstance(preds, list):
        raise RuntimeError(f"Unexpected model output: {str(out)[:400]}")
    return preds


def _predict_in_batches(endpoint: str, X: pd.DataFrame, batch_size: int = 256) -> pd.DataFrame:
    """
    Do batched inference for the entire city slice (latest dt).
    - Builds inferenceId = f"{dt}_{station_id}" deterministically.
    - Uses 'batch_size' to control payload size; tune 128~512 based on endpoint latency.
    """
    # Extract model features as floats (order must match FEATURE_COLUMNS)
    feats = X[FEATURE_COLUMNS].astype("float64")
    rows = feats.values.tolist()

    preds_all = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        preds_all.extend(_invoke_endpoint_batch(endpoint, batch))

    # Normalize per-row predictions to scalars
    def _to_scalar(v):
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, list) and len(v) == 1 and isinstance(v[0], (int, float)):
            return float(v[0])
        if isinstance(v, dict) and "yhat" in v:
            return float(v["yhat"])
        # Fallback: try numeric cast
        try:
            return float(v)
        except Exception:
            return float("nan")

    yhat = [_to_scalar(p) for p in preds_all]

    out = X[["city", "dt", "station_id"]].copy()
    out["yhat_bikes"] = yhat
    # If your model is regression-to-bikes, keep a conservative binary. Tune threshold in monitoring job.
    out["yhat_bikes_bin"] = (out["yhat_bikes"] >= 0.15).astype("float64")
    out["inferenceId"] = out["dt"].astype(str) + "_" + out["station_id"].astype(str)

    # Retain the raw response for auditing (as a compact JSON list; optional)
    out["raw"] = json.dumps(preds_all, ensure_ascii=False)

    return out[["station_id", "dt", "yhat_bikes", "yhat_bikes_bin", "inferenceId", "raw"]]


# ---------- Main entry ----------


def main():
    # 1) Read config and endpoint name
    cfg = read_env()
    city = cfg["city"]
    bucket = cfg["bucket"]
    endpoint = os.environ.get("SM_ENDPOINT", "bikeshare-staging")

    # 2) Build ONLINE features for the latest 5-min snapshot (city-wide)
    X = build_online_features(city)
    latest_dt = str(X["dt"].iloc[0])

    # 3) Batch inference
    preds = _predict_in_batches(endpoint, X, batch_size=256)

    # 4) Write the single parquet shard for this dt
    key = _prediction_key(city, latest_dt)
    _write_parquet_s3(preds, bucket, key)

    # 5) (Optional but recommended) Best-effort MSCK REPAIR for the 'inference' table
    try:
        cnx = athena_conn(
            region=cfg["region"],
            s3_staging_dir=cfg["athena_output"],
            workgroup=cfg["athena_workgroup"],
            schema_name=cfg["athena_database"],
        )
        pd.read_sql("MSCK REPAIR TABLE inference", cnx)
    except Exception:
        pass

    print(f"[predictor] wrote {len(preds)} rows to s3://{bucket}/{key}")


if __name__ == "__main__":
    # PowerShell:
    # PS> $env:SM_ENDPOINT="bikeshare-prod"; python -m src.inference.predictor
    main()
