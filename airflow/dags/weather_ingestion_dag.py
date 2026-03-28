import os
import sys
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
if AIRFLOW_HOME not in sys.path:
    sys.path.append(AIRFLOW_HOME)

from queue_defs import WEATHER_10M_QUEUE
from runtime_utils import get_airflow_setting as _get_setting, get_dw_conn_uri
from src.config import run_project_module
from schedule_defs import WEATHER_10MIN_SCHEDULE


def _get_queue_name() -> str:
    return _get_setting("AIRFLOW_QUEUE_WEATHER_10M", "AIRFLOW_QUEUE_WEATHER_10M", WEATHER_10M_QUEUE)


def _raw_bucket() -> str:
    bucket = _get_setting("BUCKET", "BUCKET", "")
    if not bucket:
        raise ValueError("BUCKET is required for weather dual-write ingestion")
    return bucket


def create_weather_staging_tables_task():
    return run_project_module(
        "src.ingest.weather_ingest",
        args=[
            "--conn-uri",
            _dw_conn_uri(),
            "--ensure-only",
        ],
        cwd=AIRFLOW_HOME,
    )


def ingest_weather_task(**context):
    api_key = _get_setting("OPENWEATHER_API_KEY", "OPENWEATHER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENWEATHER_API_KEY is required for weather ingestion")
    return run_project_module(
        "src.ingest.weather_ingest",
        args=[
            "--conn-uri",
            _dw_conn_uri(),
            "--city",
            _get_setting("CITY", "CITY", "paris"),
            "--run-id",
            context["run_id"],
            "--raw-bucket",
            _raw_bucket(),
            "--api-key",
            api_key,
            "--timeout-sec",
            _get_setting("WEATHER_HTTP_TIMEOUT_SEC", "WEATHER_HTTP_TIMEOUT_SEC", "30"),
        ],
        cwd=AIRFLOW_HOME,
    )


default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

start = pendulum.datetime(2026, 3, 1, tz="Europe/Paris")

with DAG(
    dag_id="weather_10min",
    start_date=start,
    schedule=WEATHER_10MIN_SCHEDULE,
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["weather", "openweather", "10min", "dual-write"],
):
    create_table = PythonOperator(
        task_id="create_weather_staging_tables",
        python_callable=create_weather_staging_tables_task,
        queue=_get_queue_name(),
    )
    ingest = PythonOperator(
        task_id="ingest_weather_dual_write",
        python_callable=ingest_weather_task,
        queue=_get_queue_name(),
    )
    create_table >> ingest
