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

from src.ingest.gbfs_ingest import ensure_staging_tables, ingest_station_information_to_staging, ingest_station_status_to_staging


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


def _raw_bucket() -> str:
    bucket = _get_setting("BUCKET", "BUCKET", "")
    if not bucket:
        raise ValueError("BUCKET is required for dual-write ingestion")
    return bucket


def create_staging_tables_task():
    ensure_staging_tables(conn_uri=_dw_conn_uri())


def ingest_station_information_task(**context):
    result = ingest_station_information_to_staging(
        conn_uri=_dw_conn_uri(),
        gbfs_root_url=_get_setting(
            "GBFS_BASE_URL",
            "GBFS_BASE_URL",
            "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/gbfs.json",
        ),
        run_id=context["run_id"],
        timeout_sec=int(_get_setting("GBFS_HTTP_TIMEOUT_SEC", "GBFS_HTTP_TIMEOUT_SEC", "30")),
        raw_bucket=_raw_bucket(),
        raw_city=_get_setting("GBFS_CITY", "GBFS_CITY", "paris"),
    )
    return result


def ingest_station_status_task(**context):
    result = ingest_station_status_to_staging(
        conn_uri=_dw_conn_uri(),
        gbfs_root_url=_get_setting(
            "GBFS_BASE_URL",
            "GBFS_BASE_URL",
            "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/gbfs.json",
        ),
        run_id=context["run_id"],
        timeout_sec=int(_get_setting("GBFS_HTTP_TIMEOUT_SEC", "GBFS_HTTP_TIMEOUT_SEC", "30")),
        raw_bucket=_raw_bucket(),
        raw_city=_get_setting("GBFS_CITY", "GBFS_CITY", "paris"),
    )
    return result


default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

start = pendulum.datetime(2026, 3, 1, tz="Europe/Paris")

with DAG(
    dag_id="gbfs_station_information_daily",
    start_date=start,
    schedule="0 2 * * *",
    catchup=False,
    default_args=default_args,
    tags=["gbfs", "paris", "daily"],
):
    info_create = PythonOperator(task_id="create_staging_tables", python_callable=create_staging_tables_task)
    info_ingest = PythonOperator(task_id="ingest_station_information", python_callable=ingest_station_information_task)
    info_create >> info_ingest

with DAG(
    dag_id="gbfs_station_status_5min",
    start_date=start,
    schedule="*/5 * * * *",
    catchup=False,
    default_args=default_args,
    tags=["gbfs", "paris", "5min"],
):
    status_create = PythonOperator(task_id="create_staging_tables", python_callable=create_staging_tables_task)
    status_ingest = PythonOperator(task_id="ingest_station_status", python_callable=ingest_station_status_task)
    status_create >> status_ingest
