import os
import sys
from datetime import datetime, timedelta, timezone

import pendulum
from airflow import DAG
from airflow.hooks.base import BaseHook
from airflow.models import Variable
from airflow.operators.python import PythonOperator, ShortCircuitOperator

AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
if AIRFLOW_HOME not in sys.path:
    sys.path.append(AIRFLOW_HOME)

from src.ingest.weather_ingest import ensure_weather_staging_tables, ingest_weather_dual_write, weather_snapshot_exists


def _get_setting(var_key: str, env_key: str, default_value: str) -> str:
    return Variable.get(var_key, default_var=os.getenv(env_key, default_value))


def _dw_conn_uri() -> str:
    conn_id = _get_setting("DW_CONN_ID", "DW_CONN_ID", "velib_dw")
    conn = BaseHook.get_connection(conn_id)
    uri = conn.get_uri()
    if uri.startswith("postgres://"):
        return uri.replace("postgres://", "postgresql+psycopg2://", 1)
    if uri.startswith("postgresql://"):
        return uri.replace("postgresql://", "postgresql+psycopg2://", 1)
    return uri


def _target_bucket_utc() -> datetime:
    now_utc = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    return now_utc.replace(minute=(now_utc.minute // 10) * 10)


def _raw_bucket() -> str:
    bucket = _get_setting("RAW_S3_BUCKET", "RAW_S3_BUCKET", os.getenv("BUCKET", ""))
    if not bucket:
        raise ValueError("RAW_S3_BUCKET (or BUCKET) is required for weather dual-write ingestion")
    return bucket


def create_weather_staging_tables_task():
    ensure_weather_staging_tables(conn_uri=_dw_conn_uri())


def should_ingest_weather_bucket_task() -> bool:
    city = _get_setting("WEATHER_CITY", "WEATHER_CITY", "paris")
    exists = weather_snapshot_exists(conn_uri=_dw_conn_uri(), city=city, snapshot_bucket_at_utc=_target_bucket_utc())
    return not exists


def ingest_weather_task(**context):
    api_key = _get_setting("OPENWEATHER_API_KEY", "OPENWEATHER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENWEATHER_API_KEY is required for weather ingestion")
    result = ingest_weather_dual_write(
        conn_uri=_dw_conn_uri(),
        city=_get_setting("WEATHER_CITY", "WEATHER_CITY", "paris"),
        run_id=context["run_id"],
        raw_bucket=_raw_bucket(),
        api_key=api_key,
        timeout_sec=int(_get_setting("WEATHER_HTTP_TIMEOUT_SEC", "WEATHER_HTTP_TIMEOUT_SEC", "30")),
        target_bucket_utc=_target_bucket_utc(),
    )
    return result


default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

start = pendulum.datetime(2026, 3, 1, tz="Europe/Paris")

with DAG(
    dag_id="weather_10min",
    start_date=start,
    schedule="*/10 * * * *",
    catchup=False,
    default_args=default_args,
    tags=["weather", "openweather", "10min", "dual-write"],
):
    create_table = PythonOperator(
        task_id="create_weather_staging_tables",
        python_callable=create_weather_staging_tables_task,
    )
    should_ingest = ShortCircuitOperator(
        task_id="check_not_ingested_for_target_bucket",
        python_callable=should_ingest_weather_bucket_task,
    )
    ingest = PythonOperator(
        task_id="ingest_weather_dual_write",
        python_callable=ingest_weather_task,
    )
    create_table >> should_ingest >> ingest
