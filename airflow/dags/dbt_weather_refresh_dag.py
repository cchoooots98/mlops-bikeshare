import os
import sys
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
if AIRFLOW_HOME not in sys.path:
    sys.path.append(AIRFLOW_HOME)

from dbt_thread_utils import get_dbt_threads
from queue_defs import DBT_WEATHER_POOL, WEATHER_10M_QUEUE
from runtime_utils import get_airflow_setting as _get_setting
from src.orchestration.dbt_tasks import DEFAULT_WEATHER_REFRESH_SELECT, parse_select_models, run_model_build
from external_task_utils import execution_date_fn_for_schedule
from schedule_defs import DBT_WEATHER_REFRESH_10MIN_SCHEDULE, WEATHER_10MIN_SCHEDULE


def _get_pool_name() -> str:
    return _get_setting("DBT_WEATHER_POOL", "DBT_WEATHER_POOL", DBT_WEATHER_POOL)


def _get_queue_name() -> str:
    return _get_setting("AIRFLOW_QUEUE_WEATHER_10M", "AIRFLOW_QUEUE_WEATHER_10M", WEATHER_10M_QUEUE)


def _get_weather_threads() -> int:
    return get_dbt_threads(_get_setting, "DBT_WEATHER_THREADS")


def run_dbt_weather_refresh_task():
    model_select = parse_select_models(
        _get_setting(
            "DBT_WEATHER_REFRESH_SELECT",
            "DBT_WEATHER_REFRESH_SELECT",
            " ".join(DEFAULT_WEATHER_REFRESH_SELECT),
        ),
        default_models=DEFAULT_WEATHER_REFRESH_SELECT,
    )
    summary = run_model_build(
        project_dir=_get_setting("DBT_PROJECT_DIR", "DBT_PROJECT_DIR", "dbt/bikeshare_dbt"),
        profiles_dir=_get_setting("DBT_PROFILES_DIR", "DBT_PROFILES_DIR", "dbt"),
        select_models=model_select,
        selector=None,
        threads=_get_weather_threads(),
    )
    print(f"AIRFLOW_TASK_METRIC run_dbt_weather_refresh {summary}")


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

start = pendulum.datetime(2026, 3, 1, tz="Europe/Paris")

with DAG(
    dag_id="dbt_weather_refresh_10min",
    start_date=start,
    schedule=DBT_WEATHER_REFRESH_10MIN_SCHEDULE,
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["dbt", "weather", "10min", "contract_critical"],
) as dag:
    wait_for_weather_ingest = ExternalTaskSensor(
        task_id="wait_for_weather_ingest",
        external_dag_id="weather_10min",
        external_task_id=None,
        allowed_states=["success"],
        failed_states=["failed"],
        execution_date_fn=execution_date_fn_for_schedule(WEATHER_10MIN_SCHEDULE),
        check_existence=True,
        mode="reschedule",
        poke_interval=30,
        timeout=20 * 60,
        queue=_get_queue_name(),
    )
    run_weather_refresh = PythonOperator(
        task_id="run_dbt_weather_refresh",
        python_callable=run_dbt_weather_refresh_task,
        queue=_get_queue_name(),
        pool=_get_pool_name(),
    )

    wait_for_weather_ingest >> run_weather_refresh
