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
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Reuse your existing modules
from src.features.build_features import athena_conn, read_env
from src.features.schema import FEATURE_COLUMNS
from src.inference.featurize_online import build_online_features

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
cloudwatch = boto3.client("cloudwatch", region_name="ca-central-1")
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
def _invoke_endpoint_one(endpoint: str, feature_row: list, inference_id: str) -> float:
    """
    Invoke the endpoint ONCE with a single-row payload and an explicit InferenceId.
    Returns a scalar prediction.
    """
    rt = _smr()
    # Build single-row dataframe_split payload
    payload = {
        "inputs": {
            "dataframe_split": {
                "columns": FEATURE_COLUMNS,
                "data": [feature_row],  # single row
            }
        }
    }
    resp = rt.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload).encode("utf-8"),
        # This is the KEY: attach the per-row InferenceId so DataCapture records it
        InferenceId=inference_id,
    )
    body = resp["Body"].read().decode("utf-8", errors="ignore")
    out = json.loads(body) if body.strip().startswith(("{", "[")) else body
    preds = out.get("predictions", out) if isinstance(out, dict) else out

    # Normalize output to a scalar float
    if isinstance(preds, list):
        v = preds[0] if len(preds) == 1 else preds
    else:
        v = preds
    if isinstance(v, list) and len(v) == 1 and isinstance(v[0], (int, float)):
        v = v[0]
    try:
        return float(v if not isinstance(v, dict) else v.get("yhat", "nan"))
    except Exception:
        return float("nan")


def _predict_rowwise_threaded(endpoint: str, X: pd.DataFrame, max_workers: int = 16) -> pd.DataFrame:
    """
    Row-wise inference with per-row InferenceId and concurrency.
    - inferenceId format: f"{dt}_{station_id}"
    - max_workers controls parallelism; tune per endpoint capacity and CPS limits.
    """
    # Prepare feature matrix (order must match FEATURE_COLUMNS)
    feats = X[FEATURE_COLUMNS].astype("float64").values.tolist()

    # Precompute the per-row inferenceId (this is also what we'll write to parquet)
    ids = (X["dt"].astype(str) + "_" + X["station_id"].astype(str)).tolist()

    yhat = [None] * len(feats)

    # Threaded dispatch
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_invoke_endpoint_one, endpoint, feats[i], ids[i]): i for i in range(len(feats))}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                yhat[idx] = fut.result()
            except Exception:
                # On failure, mark NaN; you can also collect errors if needed
                yhat[idx] = float("nan")

    out = X[["city", "dt", "station_id"]].copy()
    out["yhat_bikes"] = yhat
    out["yhat_bikes_bin"] = (out["yhat_bikes"] >= 0.15).astype("float64")
    out["inferenceId"] = ids
    # Keep a tiny JSON string per row if you like; here we omit raw batch responses
    out["raw"] = ""  # optional: leave empty or store per-row if you want
    return out[["station_id", "dt", "yhat_bikes", "yhat_bikes_bin", "inferenceId", "raw"]]


def publish_heartbeat(endpoint_name: str, city: str):
    cloudwatch.put_metric_data(
        Namespace="Bikeshare/Model",
        MetricData=[{
            "MetricName": "PredictionHeartbeat",
            "Value": 1,
            "Unit": "Count",
            "Dimensions": [
                {"Name": "EndpointName", "Value": endpoint_name},
                {"Name": "City", "Value": city}
            ]
        }]
    )


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
    # preds = _predict_in_batches(endpoint, X, batch_size=512)
    preds = _predict_rowwise_threaded(endpoint, X, max_workers=16)

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
    publish_heartbeat(endpoint_name=endpoint, city=city)


if __name__ == "__main__":
    # PowerShell:
    # PS> $env:SM_ENDPOINT="bikeshare-prod"; python -m src.inference.predictor
    main()
