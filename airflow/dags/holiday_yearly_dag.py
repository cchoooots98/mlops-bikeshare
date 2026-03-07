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

from src.ingest.holidays_ingest import ingest_holidays_year


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


def ingest_holidays_year_task(**context):
    logical_year = context["logical_date"].in_timezone("Europe/Paris").year
    dag_run = context.get("dag_run")
    conf = (dag_run.conf or {}) if dag_run else {}
    year = int(conf.get("year", logical_year))

    result = ingest_holidays_year(
        conn_uri=_dw_conn_uri(),
        year=year,
        run_id=context["run_id"],
        country_code=_get_setting("HOLIDAY_COUNTRY_CODE", "HOLIDAY_COUNTRY_CODE", "FR"),
        timeout_sec=int(_get_setting("HOLIDAY_HTTP_TIMEOUT_SEC", "HOLIDAY_HTTP_TIMEOUT_SEC", "30")),
        raw_bucket=_get_setting("RAW_S3_BUCKET", "RAW_S3_BUCKET", os.getenv("BUCKET", "")),
    )
    return result


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

start = pendulum.datetime(2026, 1, 1, tz="Europe/Paris")

with DAG(
    dag_id="holiday_yearly",
    start_date=start,
    schedule="0 2 1 1 *",
    catchup=False,
    default_args=default_args,
    tags=["holidays", "yearly", "dim_date"],
):
    ingest = PythonOperator(
        task_id="ingest_holidays_and_update_dim_date",
        python_callable=ingest_holidays_year_task,
    )
