import io
from datetime import datetime, timedelta, timezone
from typing import List

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import prediction_key, prediction_prefix, quality_key
from src.features.postgres_store import PostgresFeatureConfig, create_pg_engine, load_training_actuals_for_dt
from src.model_target import PredictionTargetSpec, target_spec_from_name, target_spec_from_predict_bikes
from src.config import load_runtime_settings

MATURITY_MINUTES = 30
BACKFILL_MINUTES = 120
STEP_MINUTES = 5


def _s3():
    return boto3.client("s3")


def _object_exists(bucket: str, key: str) -> bool:
    try:
        _s3().head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def _read_parquet_s3(bucket: str, key: str) -> pd.DataFrame:
    try:
        obj = _s3().get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
        table = pq.read_table(io.BytesIO(data))
        return table.to_pandas()
    except Exception:
        return pd.DataFrame()


def _write_parquet_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    _s3().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


def _make_candidate_dts(latest_dt_str: str, now_utc: datetime) -> List[str]:
    base = datetime.strptime(latest_dt_str, "%Y-%m-%d-%H-%M").replace(tzinfo=timezone.utc)
    out = []
    steps = BACKFILL_MINUTES // STEP_MINUTES
    for step in range(0, steps + 1):
        dt_value = base - timedelta(minutes=STEP_MINUTES * step)
        if now_utc >= dt_value + timedelta(minutes=MATURITY_MINUTES):
            out.append(dt_value.strftime("%Y-%m-%d-%H-%M"))
    return out


def _list_prediction_dts(bucket: str, city: str, target_name: str) -> list[str]:
    prefix = prediction_prefix(city, target_name)
    paginator = _s3().get_paginator("list_objects_v2")
    dts = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.endswith("/predictions.parquet"):
                continue
            try:
                dt_value = key.split("/dt=", 1)[1].split("/", 1)[0]
            except IndexError:
                continue
            dts.add(dt_value)
    return sorted(dts)


def _resolve_target_spec(preds: pd.DataFrame, default_predict_bikes: bool) -> PredictionTargetSpec:
    if "prediction_target" in preds.columns:
        target_values = preds["prediction_target"].dropna().astype(str).str.lower().unique().tolist()
        if len(target_values) == 1:
            return target_spec_from_name(target_values[0])
        if len(target_values) > 1:
            raise ValueError(f"prediction shard must contain exactly one prediction_target, got {target_values}")
    return target_spec_from_predict_bikes(default_predict_bikes)


def _load_actuals_for_dt(
    engine,
    pg_config: PostgresFeatureConfig,
    city: str,
    pred_dt: str,
    target_spec: PredictionTargetSpec,
) -> pd.DataFrame:
    actuals = load_training_actuals_for_dt(
        engine,
        pg_config,
        city,
        pred_dt,
        predict_bikes=target_spec.predict_bikes,
    )
    if actuals.empty:
        return pd.DataFrame(
            columns=["station_id", target_spec.label_column, target_spec.actual_t30_column, "dt_plus30", "min_inventory_within_30m"]
        )
    dt_plus30 = (datetime.strptime(pred_dt, "%Y-%m-%d-%H-%M") + timedelta(minutes=30)).strftime("%Y-%m-%d-%H-%M")
    actuals = actuals.dropna(subset=[target_spec.label_column]).copy()
    actuals["dt_plus30"] = dt_plus30
    actuals[target_spec.actual_t30_column] = actuals[target_spec.paired_target_column]
    actuals["min_inventory_within_30m"] = pd.NA
    return actuals[
        ["station_id", "min_inventory_within_30m", target_spec.actual_t30_column, target_spec.label_column, "dt_plus30"]
    ]


def _build_quality_rows(
    preds: pd.DataFrame,
    acts: pd.DataFrame,
    dt_pred: str,
    target_spec: PredictionTargetSpec,
) -> pd.DataFrame:
    if "prediction_target" not in preds.columns:
        preds = preds.assign(prediction_target=target_spec.target_name)
    return (
        preds.merge(acts, on="station_id", how="inner")
        .assign(dt=lambda frame: dt_pred)
        .loc[
            :,
            [
                "station_id",
                "dt",
                "dt_plus30",
                "prediction_target",
                target_spec.score_column,
                target_spec.score_bin_column,
                target_spec.label_column,
                "min_inventory_within_30m",
                target_spec.actual_t30_column,
                "inferenceId",
            ],
        ]
    )


def main():
    settings = load_runtime_settings()
    if not settings.bucket:
        raise ValueError("missing required runtime setting: BUCKET")

    pg_config = PostgresFeatureConfig(
        pg_host=settings.pg_host,
        pg_port=settings.pg_port,
        pg_db=settings.pg_db,
        pg_user=settings.pg_user,
        pg_password=settings.pg_password,
        pg_schema=settings.pg_schema,
        training_table=settings.training_feature_table,
        online_table=settings.online_feature_table,
    )
    engine = create_pg_engine(pg_config)

    prediction_dts = _list_prediction_dts(settings.bucket, settings.city, settings.target_name)
    if not prediction_dts:
        print("[quality] no predictions found; nothing to backfill")
        return

    latest_dt = prediction_dts[-1]
    candidates = _make_candidate_dts(latest_dt, datetime.now(timezone.utc))
    shards_written = 0

    for dt_pred in candidates:
        pred_key = prediction_key(settings.city, dt_pred, settings.target_name)
        qual_key = quality_key(settings.city, dt_pred, settings.target_name)
        if _object_exists(settings.bucket, qual_key) or not _object_exists(settings.bucket, pred_key):
            continue

        preds = _read_parquet_s3(settings.bucket, pred_key)
        if preds.empty:
            continue

        target_spec = _resolve_target_spec(preds, settings.predict_bikes)
        pred_key = prediction_key(settings.city, dt_pred, target_spec.target_name)
        qual_key = quality_key(settings.city, dt_pred, target_spec.target_name)
        acts = _load_actuals_for_dt(engine, pg_config, settings.city, dt_pred, target_spec)
        if acts.empty:
            continue

        joined = _build_quality_rows(preds, acts, dt_pred, target_spec)
        if joined.empty:
            continue

        _write_parquet_s3(joined, settings.bucket, qual_key)
        shards_written += 1

    print(f"[quality] wrote {shards_written} shard(s).")


if __name__ == "__main__":
    main()
