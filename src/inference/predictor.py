import io
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import prediction_key
from src.features.schema import NULLABLE_NAN_PRESERVED_COLUMNS, NULLABLE_ZERO_FILL_COLUMNS
from src.monitoring.metrics.metrics_helper import publish_heartbeat
from src.model_package import load_package_manifest, resolve_active_package_dir
from src.inference.featurize_online import build_online_features
from src.model_target import PredictionTargetSpec, target_spec_from_metadata
from src.config import load_runtime_settings

DEFAULT_MAX_WORKERS = 16
DEFAULT_MAX_ATTEMPTS = 3
BASE_RETRY_BACKOFF_SECONDS = 0.5
NULLABLE_NAN_PRESERVED_COLUMN_SET = set(NULLABLE_NAN_PRESERVED_COLUMNS)
NULLABLE_ZERO_FILL_COLUMN_SET = set(NULLABLE_ZERO_FILL_COLUMNS)


@dataclass(frozen=True)
class PredictionFailure:
    inference_id: str
    error: str


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


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
    if not isinstance(output, (dict, list)):
        raise ValueError(f"endpoint returned non-JSON prediction payload: {type(output).__name__}")
    preds = output.get("predictions", output) if isinstance(output, dict) else output
    if isinstance(preds, list):
        value = preds[0] if len(preds) == 1 else preds
    else:
        value = preds
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (int, float)):
        value = value[0]
    if isinstance(value, dict):
        if "yhat" not in value:
            raise ValueError(f"endpoint returned dict prediction without 'yhat': {value}")
        value = value["yhat"]
    if isinstance(value, list):
        raise ValueError(f"endpoint returned multiple predictions for a single row: {value}")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"endpoint returned non-finite prediction: {value}")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"endpoint returned prediction outside [0, 1]: {score}")
    return score


def _validate_feature_row(feature_row: list[float], feature_columns: list[str], inference_id: str) -> None:
    invalid_columns: list[str] = []
    for index, value in enumerate(feature_row):
        if not _is_finite_number(value) and feature_columns[index] not in NULLABLE_NAN_PRESERVED_COLUMN_SET:
            invalid_columns.append(feature_columns[index])
    if invalid_columns:
        preview = ", ".join(invalid_columns[:5])
        suffix = "..." if len(invalid_columns) > 5 else ""
        raise ValueError(
            f"inference_id={inference_id} contains non-finite feature values; columns={preview}{suffix}"
        )


def _sanitize_feature_frame(
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
    inference_ids: list[str],
) -> pd.DataFrame:
    zero_filled: list[str] = []
    nan_preserved: list[str] = []
    hard_failures: list[str] = []

    for column in feature_columns:
        invalid_mask = ~feature_frame[column].map(_is_finite_number)
        if not invalid_mask.any():
            continue

        bad_positions = [index for index, is_bad in enumerate(invalid_mask.tolist()) if is_bad]
        bad_count = len(bad_positions)
        if column in NULLABLE_ZERO_FILL_COLUMN_SET:
            feature_frame.loc[invalid_mask, column] = 0.0
            zero_filled.append(f"{column}={bad_count}")
            continue

        if column in NULLABLE_NAN_PRESERVED_COLUMN_SET:
            feature_frame.loc[invalid_mask, column] = math.nan
            nan_preserved.append(f"{column}={bad_count}")
            continue

        bad_examples = ", ".join(inference_ids[index] for index in bad_positions[:3])
        hard_failures.append(f"{column}={bad_count} rows (examples: {bad_examples})")

    if zero_filled:
        preview = ", ".join(zero_filled[:5])
        suffix = "..." if len(zero_filled) > 5 else ""
        print(
            "[predictor] coerced non-finite nullable feature values to 0.0 "
            f"for columns: {preview}{suffix}"
        )

    if nan_preserved:
        preview = ", ".join(nan_preserved[:5])
        suffix = "..." if len(nan_preserved) > 5 else ""
        print(
            "[predictor] coerced non-finite nullable feature values to NaN "
            f"for columns: {preview}{suffix}"
        )

    if hard_failures:
        raise ValueError(
            "non-finite values found in non-nullable feature columns: "
            + "; ".join(hard_failures[:5])
        )

    return feature_frame


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
    if not body.strip().startswith(("{", "[")):
        raise ValueError(f"endpoint returned non-JSON body for {inference_id}: {body[:200]}")
    out = json.loads(body)
    return _coerce_prediction_value(out)


def _invoke_endpoint_with_retry(
    endpoint: str,
    feature_row: list[float],
    inference_id: str,
    feature_columns: list[str],
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_retry_backoff_seconds: float = BASE_RETRY_BACKOFF_SECONDS,
) -> float:
    _validate_feature_row(feature_row, feature_columns, inference_id)
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return _invoke_endpoint_one(endpoint, feature_row, inference_id, feature_columns)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            sleep_seconds = base_retry_backoff_seconds * (2 ** (attempt - 1))
            print(
                f"[predictor] endpoint={endpoint} inference_id={inference_id} "
                f"attempt={attempt}/{max_attempts} failed with {type(exc).__name__}: {str(exc)[:180]} "
                f"retrying_after={sleep_seconds:.2f}s"
            )
            time.sleep(sleep_seconds)

    assert last_exc is not None
    raise RuntimeError(
        f"inference_id={inference_id} exhausted {max_attempts} attempts with "
        f"{type(last_exc).__name__}: {str(last_exc)[:200]}"
    )


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
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> pd.DataFrame:
    ids = (X["dt"].astype(str) + "_" + X["station_id"].astype(str)).tolist()
    feature_frame = _sanitize_feature_frame(
        X[feature_columns].astype("float64").reset_index(drop=True),
        feature_columns,
        ids,
    )
    feats = feature_frame.values.tolist()
    yhat = [None] * len(feats)
    failures: list[PredictionFailure] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _invoke_endpoint_with_retry,
                endpoint,
                feats[index],
                ids[index],
                feature_columns,
                max_attempts=max_attempts,
            ): index
            for index in range(len(feats))
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                yhat[index] = future.result()
            except Exception as exc:
                failures.append(
                    PredictionFailure(
                        inference_id=ids[index],
                        error=f"{type(exc).__name__}: {str(exc)}"[:320],
                    )
                )

    if failures:
        examples = "; ".join(f"{item.inference_id}: {item.error}" for item in failures[:3])
        raise RuntimeError(
            f"endpoint={endpoint} failed predictions for {len(failures)}/{len(feats)} rows; examples: {examples}"
        )

    valid_prediction_count = sum(
        1 for value in yhat if value is not None and math.isfinite(float(value))
    )
    if valid_prediction_count == 0:
        raise RuntimeError(f"endpoint={endpoint} produced all predictions invalid for {len(feats)} rows")

    out = X[["city", "station_id", "dt"]].copy()
    out["prediction_target"] = target_spec.target_name
    out[target_spec.score_column] = yhat
    out[target_spec.score_bin_column] = (out[target_spec.score_column] >= threshold).astype("float64")
    out["inferenceId"] = ids
    out["raw"] = ""
    print(
        f"[predictor] endpoint={endpoint} target={target_spec.target_name} "
        f"valid_predictions={valid_prediction_count}/{len(feats)}"
    )
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
        max_workers=DEFAULT_MAX_WORKERS,
        max_attempts=DEFAULT_MAX_ATTEMPTS,
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
