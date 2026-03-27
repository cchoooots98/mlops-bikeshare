import os
import sys
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.hooks.base import BaseHook
from airflow.models import Variable
from airflow.operators.python import PythonOperator

AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
if AIRFLOW_HOME not in sys.path:
    sys.path.append(AIRFLOW_HOME)

from src.config import run_project_module
from schedule_defs import GBFS_STATION_INFORMATION_DAILY_SCHEDULE, GBFS_STATION_STATUS_5MIN_SCHEDULE


def _get_setting(var_key: str, env_key: str, default_value: str) -> str:
    return Variable.get(var_key, default_var=os.getenv(env_key, default_value))


def _tier1_queue() -> str:
    return _get_setting("AIRFLOW_TIER1_QUEUE", "AIRFLOW_TIER1_QUEUE", "tier1")


def _tier2_queue() -> str:
    return _get_setting("AIRFLOW_TIER2_QUEUE", "AIRFLOW_TIER2_QUEUE", "tier2")


def _dw_conn_uri() -> str:
    conn_id = _get_setting("DW_CONN_ID", "DW_CONN_ID", "velib_dw")
    conn = BaseHook.get_connection(conn_id)
    uri = conn.get_uri()
    if uri.startswith("postgres://"):
        return uri.replace("postgres://", "postgresql+psycopg2://", 1)
    if uri.startswith("postgresql://"):
        return uri.replace("postgresql://", "postgresql+psycopg2://", 1)
    return uri


def _raw_bucket() -> str:
    bucket = _get_setting("BUCKET", "BUCKET", "")
    if not bucket:
        raise ValueError("BUCKET is required for dual-write ingestion")
    return bucket


def create_staging_tables_task():
    return run_project_module(
        "src.ingest.gbfs_ingest",
        args=[
            "--conn-uri",
            _dw_conn_uri(),
            "--ensure-only",
        ],
        cwd=AIRFLOW_HOME,
    )


def ingest_station_information_task(**context):
    return run_project_module(
        "src.ingest.gbfs_ingest",
        args=[
            "--conn-uri",
            _dw_conn_uri(),
            "--gbfs-root-url",
            _get_setting(
                "GBFS_BASE_URL",
                "GBFS_BASE_URL",
                "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/gbfs.json",
            ),
            "--run-id",
            context["run_id"],
            "--timeout-sec",
            _get_setting("GBFS_HTTP_TIMEOUT_SEC", "GBFS_HTTP_TIMEOUT_SEC", "30"),
            "--raw-bucket",
            _raw_bucket(),
            "--city",
            _get_setting("GBFS_CITY", "GBFS_CITY", "paris"),
            "--feed",
            "station_information",
        ],
        cwd=AIRFLOW_HOME,
    )


def ingest_station_status_task(**context):
    return run_project_module(
        "src.ingest.gbfs_ingest",
        args=[
            "--conn-uri",
            _dw_conn_uri(),
            "--gbfs-root-url",
            _get_setting(
                "GBFS_BASE_URL",
                "GBFS_BASE_URL",
                "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/gbfs.json",
            ),
            "--run-id",
            context["run_id"],
            "--timeout-sec",
            _get_setting("GBFS_HTTP_TIMEOUT_SEC", "GBFS_HTTP_TIMEOUT_SEC", "30"),
            "--raw-bucket",
            _raw_bucket(),
            "--city",
            _get_setting("GBFS_CITY", "GBFS_CITY", "paris"),
            "--feed",
            "station_status",
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
    dag_id="gbfs_station_information_daily",
    start_date=start,
    schedule=GBFS_STATION_INFORMATION_DAILY_SCHEDULE,
    catchup=False,
    default_args=default_args,
    tags=["gbfs", "paris", "daily"],
):
    info_create = PythonOperator(
        task_id="create_staging_tables",
        python_callable=create_staging_tables_task,
        queue=_tier2_queue(),
    )
    info_ingest = PythonOperator(
        task_id="ingest_station_information",
        python_callable=ingest_station_information_task,
        queue=_tier2_queue(),
    )
    info_create >> info_ingest

with DAG(
    dag_id="gbfs_station_status_5min",
    start_date=start,
    schedule=GBFS_STATION_STATUS_5MIN_SCHEDULE,
    catchup=False,
    default_args=default_args,
    tags=["gbfs", "paris", "5min"],
):
    status_create = PythonOperator(
        task_id="create_staging_tables",
        python_callable=create_staging_tables_task,
        queue=_tier1_queue(),
    )
    status_ingest = PythonOperator(
        task_id="ingest_station_status",
        python_callable=ingest_station_status_task,
        queue=_tier1_queue(),
    )
    status_create >> status_ingest
