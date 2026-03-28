import os
import sys
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
if AIRFLOW_HOME not in sys.path:
    sys.path.append(AIRFLOW_HOME)

from queue_defs import DAILY_SIDECAR_QUEUE
from runtime_utils import get_airflow_setting as _get_setting, get_dw_conn_uri
from src.config import run_project_module
from schedule_defs import HOLIDAY_YEARLY_SCHEDULE


def _raw_bucket() -> str:
    bucket = _get_setting("BUCKET", "BUCKET", "")
    if not bucket:
        raise ValueError("BUCKET is required for holiday dual-write ingestion")
    return bucket


def _get_queue_name() -> str:
    return _get_setting("AIRFLOW_QUEUE_DAILY_SIDECAR", "AIRFLOW_QUEUE_DAILY_SIDECAR", DAILY_SIDECAR_QUEUE)


def ingest_holidays_year_task(**context):
    logical_year = context["logical_date"].in_timezone("Europe/Paris").year
    dag_run = context.get("dag_run")
    conf = (dag_run.conf or {}) if dag_run else {}
    year = int(conf.get("year", logical_year))
    return run_project_module(
        "src.ingest.holidays_ingest",
        args=[
            "--year",
            str(year),
            "--conn-uri",
            _dw_conn_uri(),
            "--run-id",
            context["run_id"],
            "--country-code",
            _get_setting("HOLIDAY_COUNTRY_CODE", "HOLIDAY_COUNTRY_CODE", "FR"),
            "--timeout-sec",
            _get_setting("HOLIDAY_HTTP_TIMEOUT_SEC", "HOLIDAY_HTTP_TIMEOUT_SEC", "30"),
            "--raw-bucket",
            _raw_bucket(),
        ],
        cwd=AIRFLOW_HOME,
    )


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

start = pendulum.datetime(2026, 1, 1, tz="Europe/Paris")

with DAG(
    dag_id="holiday_yearly",
    start_date=start,
    schedule=HOLIDAY_YEARLY_SCHEDULE,
    catchup=False,
    default_args=default_args,
    tags=["holidays", "yearly", "dual-write"],
):
    ingest = PythonOperator(
        task_id="ingest_holidays_to_staging",
        python_callable=ingest_holidays_year_task,
        queue=_get_queue_name(),
    )
