import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import prediction_key
from src.monitoring.metrics.metrics_helper import publish_heartbeat
from src.model_package import load_package_manifest, resolve_active_package_dir
from src.inference.featurize_online import build_online_features
from src.model_target import PredictionTargetSpec, target_spec_from_metadata
from src.config import load_runtime_settings


def _s3():
    return boto3.client("s3")


def _smr():
    return boto3.client("sagemaker-runtime")


def _write_parquet_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    _s3().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


def build_endpoint_payload(feature_row: list[float], feature_columns: list[str]) -> dict:
    return {"inputs": {"dataframe_split": {"columns": feature_columns, "data": [feature_row]}}}


def _coerce_prediction_value(output) -> float:
    preds = output.get("predictions", output) if isinstance(output, dict) else output
    if isinstance(preds, list):
        value = preds[0] if len(preds) == 1 else preds
    else:
        value = preds
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (int, float)):
        value = value[0]
    try:
        return float(value if not isinstance(value, dict) else value.get("yhat", "nan"))
    except Exception:
        return float("nan")


def _invoke_endpoint_one(endpoint: str, feature_row: list[float], inference_id: str, feature_columns: list[str]) -> float:
    payload = build_endpoint_payload(feature_row, feature_columns)
    resp = _smr().invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload).encode("utf-8"),
        InferenceId=inference_id,
    )
    body = resp["Body"].read().decode("utf-8", errors="ignore")
    out = json.loads(body) if body.strip().startswith(("{", "[")) else body
    return _coerce_prediction_value(out)


def load_prediction_manifest(
    *,
    model_package_dir: str | None = None,
    deployment_state_path: str | None = None,
) -> dict:
    package_dir = resolve_active_package_dir(
        model_package_dir=model_package_dir,
        deployment_state_path=deployment_state_path,
    )
    manifest = load_package_manifest(package_dir)
    threshold = manifest.get("best_threshold")
    if threshold in {None, ""}:
        raise ValueError(f"package manifest missing best_threshold: {package_dir}")
    feature_columns = manifest.get("feature_columns", [])
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError(f"package manifest missing feature_columns: {package_dir}")
    duplicates = sorted({column for column in feature_columns if feature_columns.count(column) > 1})
    if duplicates:
        raise ValueError(f"package feature_columns must be unique: {duplicates}")
    target_spec = target_spec_from_metadata(manifest)
    return {
        "package_dir": str(package_dir),
        "predict_bikes": target_spec.predict_bikes,
        "target_name": target_spec.target_name,
        "label": target_spec.label_column,
        "label_column": target_spec.label_column,
        "paired_target_column": target_spec.paired_target_column,
        "score_column": target_spec.score_column,
        "score_bin_column": target_spec.score_bin_column,
        "actual_t30_column": target_spec.actual_t30_column,
        "best_threshold": float(threshold),
        "model_name": manifest.get("model_name"),
        "feature_source": manifest.get("feature_source"),
        "feature_columns": feature_columns,
    }


def _predict_rowwise_threaded(
    endpoint: str,
    X: pd.DataFrame,
    target_spec: PredictionTargetSpec,
    threshold: float,
    feature_columns: list[str],
    max_workers: int = 16,
) -> pd.DataFrame:
    feats = X[feature_columns].astype("float64").values.tolist()
    ids = (X["dt"].astype(str) + "_" + X["station_id"].astype(str)).tolist()
    yhat = [None] * len(feats)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_invoke_endpoint_one, endpoint, feats[index], ids[index], feature_columns): index
            for index in range(len(feats))
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                yhat[index] = future.result()
            except Exception:
                yhat[index] = float("nan")

    out = X[["city", "station_id", "dt"]].copy()
    out["prediction_target"] = target_spec.target_name
    out[target_spec.score_column] = yhat
    out[target_spec.score_bin_column] = (out[target_spec.score_column] >= threshold).astype("float64")
    out["inferenceId"] = ids
    out["raw"] = ""
    return out


def main():
    settings = load_runtime_settings()
    if not settings.bucket:
        raise ValueError("missing required runtime setting: BUCKET")

    metadata = load_prediction_manifest(
        model_package_dir=settings.model_package_dir,
        deployment_state_path=settings.deployment_state_path,
    )
    endpoint = settings.sm_endpoint
    city = settings.city
    features = build_online_features(city, feature_columns=metadata["feature_columns"])
    target_spec = target_spec_from_metadata(metadata)
    predictions = _predict_rowwise_threaded(
        endpoint,
        features,
        target_spec=target_spec,
        threshold=metadata["best_threshold"],
        feature_columns=metadata["feature_columns"],
        max_workers=16,
    )

    for dt_value, shard in predictions.groupby("dt", sort=True):
        key = prediction_key(city, str(dt_value), metadata["target_name"])
        _write_parquet_s3(shard.drop(columns=["city"]), settings.bucket, key)
        print(f"[predictor] wrote {len(shard)} rows to s3://{settings.bucket}/{key}")

    publish_heartbeat(
        endpoint=endpoint,
        city=city,
        target_name=metadata["target_name"],
        environment=settings.serving_environment,
    )


if __name__ == "__main__":
    main()
