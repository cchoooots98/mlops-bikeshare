import os
import sys
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
if AIRFLOW_HOME not in sys.path:
    sys.path.append(AIRFLOW_HOME)

from dbt_thread_utils import get_dbt_threads
from queue_defs import DAILY_SIDECAR_QUEUE, DBT_SIDECAR_POOL
from runtime_utils import get_airflow_setting as _get_setting
from src.orchestration.dbt_tasks import (
    DEFAULT_QUALITY_TEST_SELECTOR,
    DEFAULT_SOURCE_FRESHNESS_SELECT,
    parse_selector,
    parse_select_models,
    run_dbt_source_freshness,
    run_quality_tests,
)
from schedule_defs import DBT_QUALITY_HOURLY_SCHEDULE


def _get_pool_name() -> str:
    return _get_setting("DBT_SIDECAR_POOL", "DBT_SIDECAR_POOL", DBT_SIDECAR_POOL)


def _get_queue_name() -> str:
    return _get_setting("AIRFLOW_QUEUE_DAILY_SIDECAR", "AIRFLOW_QUEUE_DAILY_SIDECAR", DAILY_SIDECAR_QUEUE)


def _get_sidecar_threads() -> int:
    return get_dbt_threads(_get_setting, "DBT_SIDECAR_THREADS")


def _build_quality_test_vars(context: dict) -> dict[str, object]:
    return {
        "test_window_end_utc": context["data_interval_end"].astimezone(pendulum.UTC).isoformat(),
        "test_window_lookback_hours": int(
            _get_setting("DBT_QUALITY_LOOKBACK_HOURS", "DBT_QUALITY_LOOKBACK_HOURS", "72"),
        ),
    }


def run_dbt_source_freshness_task():
    summary = run_dbt_source_freshness(
        project_dir=_get_setting("DBT_PROJECT_DIR", "DBT_PROJECT_DIR", "dbt/bikeshare_dbt"),
        profiles_dir=_get_setting("DBT_PROFILES_DIR", "DBT_PROFILES_DIR", "dbt"),
        select_models=parse_select_models(
            _get_setting("DBT_SOURCE_FRESHNESS_SELECT", "DBT_SOURCE_FRESHNESS_SELECT", ""),
            default_models=DEFAULT_SOURCE_FRESHNESS_SELECT,
        ),
        threads=_get_sidecar_threads(),
    )
    print(f"AIRFLOW_TASK_METRIC run_dbt_source_freshness {summary}")


def run_dbt_quality_tests_task(**context):
    raw_test_select = _get_setting(
        "DBT_QUALITY_TEST_SELECT",
        "DBT_QUALITY_TEST_SELECT",
        "",
    )
    test_select = parse_select_models(raw_test_select, default_models=[]) if raw_test_select.strip() else None
    test_selector = parse_selector(
        _get_setting(
            "DBT_QUALITY_TEST_SELECTOR",
            "DBT_QUALITY_TEST_SELECTOR",
            DEFAULT_QUALITY_TEST_SELECTOR,
        ),
        default_selector=DEFAULT_QUALITY_TEST_SELECTOR,
    )
    summary = run_quality_tests(
        project_dir=_get_setting("DBT_PROJECT_DIR", "DBT_PROJECT_DIR", "dbt/bikeshare_dbt"),
        profiles_dir=_get_setting("DBT_PROFILES_DIR", "DBT_PROFILES_DIR", "dbt"),
        select_models=test_select,
        selector=None if test_select else test_selector,
        threads=_get_sidecar_threads(),
        dbt_vars=_build_quality_test_vars(context),
    )
    print(f"AIRFLOW_TASK_METRIC run_dbt_quality_tests {summary}")


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

start = pendulum.datetime(2026, 3, 1, tz="Europe/Paris")

with DAG(
    dag_id="dbt_quality_hourly",
    start_date=start,
    schedule=DBT_QUALITY_HOURLY_SCHEDULE,
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["dbt", "quality", "hourly", "contract_critical"],
) as dag:
    source_freshness = PythonOperator(
        task_id="run_dbt_source_freshness",
        python_callable=run_dbt_source_freshness_task,
        queue=_get_queue_name(),
        pool=_get_pool_name(),
    )
    quality_tests = PythonOperator(
        task_id="run_dbt_quality_tests",
        python_callable=run_dbt_quality_tests_task,
        queue=_get_queue_name(),
        pool=_get_pool_name(),
    )

    source_freshness >> quality_tests
