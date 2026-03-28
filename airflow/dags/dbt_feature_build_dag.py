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
from queue_defs import CORE_5M_QUEUE
from runtime_utils import get_airflow_setting as _get_setting
from src.orchestration.dbt_tasks import (
    DEFAULT_FEATURE_BUILD_SELECTOR,
    DEFAULT_FEATURE_BUILD_SELECT,
    DEFAULT_FEATURE_TEST_SELECTOR,
    parse_bool,
    parse_selector,
    parse_select_models,
    run_feature_model_build,
    run_feature_smoke_tests,
)
from external_task_utils import execution_date_fn_for_schedule
from schedule_defs import DBT_FEATURE_BUILD_5MIN_SCHEDULE, DBT_STATION_STATUS_HOTPATH_5MIN_SCHEDULE


def _get_pool_name() -> str:
    return _get_setting("DBT_FEATURE_POOL", "DBT_FEATURE_POOL", "dbt_feature_pool")


def _get_queue_name() -> str:
    return _get_setting("AIRFLOW_QUEUE_CORE_5M", "AIRFLOW_QUEUE_CORE_5M", CORE_5M_QUEUE)


def _build_feature_model_vars(context: dict) -> dict[str, object]:
    return {
        "feature_window_end_utc": context["data_interval_end"].astimezone(pendulum.UTC).isoformat(),
        "feature_rebuild_lookback_minutes": int(
            _get_setting("DBT_FEATURE_REBUILD_LOOKBACK_MINUTES", "DBT_FEATURE_REBUILD_LOOKBACK_MINUTES", "120"),
        ),
    }


def _build_feature_test_vars(context: dict) -> dict[str, object]:
    return {
        "test_window_end_utc": context["data_interval_end"].astimezone(pendulum.UTC).isoformat(),
        "test_window_lookback_hours": int(
            _get_setting("DBT_FEATURE_SMOKE_LOOKBACK_HOURS", "DBT_FEATURE_SMOKE_LOOKBACK_HOURS", "72"),
        ),
    }


def _get_feature_threads() -> int:
    return get_dbt_threads(_get_setting, "DBT_FEATURE_THREADS")


def run_dbt_feature_build_task(**context):
    raw_build_select = _get_setting(
        "DBT_FEATURE_BUILD_SELECT",
        "DBT_FEATURE_BUILD_SELECT",
        "",
    )
    model_select = parse_select_models(raw_build_select, default_models=[]) if raw_build_select.strip() else None
    build_selector = parse_selector(
        _get_setting(
            "DBT_FEATURE_BUILD_SELECTOR",
            "DBT_FEATURE_BUILD_SELECTOR",
            DEFAULT_FEATURE_BUILD_SELECTOR,
        ),
        default_selector=DEFAULT_FEATURE_BUILD_SELECTOR,
    )
    threads = _get_feature_threads()
    summary = run_feature_model_build(
        project_dir=_get_setting("DBT_PROJECT_DIR", "DBT_PROJECT_DIR", "dbt/bikeshare_dbt"),
        profiles_dir=_get_setting("DBT_PROFILES_DIR", "DBT_PROFILES_DIR", "dbt"),
        select_models=model_select,
        selector=None if model_select else build_selector,
        threads=threads,
        dbt_vars=_build_feature_model_vars(context),
    )
    print(f"AIRFLOW_TASK_METRIC run_dbt_feature_build {summary}")


def run_dbt_feature_smoke_tests_task(**context):
    if parse_bool(_get_setting("DBT_FEATURE_BUILD_SKIP_TESTS", "DBT_FEATURE_BUILD_SKIP_TESTS", "false")):
        summary = {"skipped": True}
        print(f"AIRFLOW_TASK_METRIC run_dbt_feature_smoke_tests {summary}")
        return
    raw_test_select = _get_setting(
        "DBT_FEATURE_BUILD_TEST_SELECT",
        "DBT_FEATURE_BUILD_TEST_SELECT",
        "",
    )
    test_select = parse_select_models(raw_test_select, default_models=[]) if raw_test_select.strip() else None
    test_selector = parse_selector(
        _get_setting(
            "DBT_FEATURE_BUILD_TEST_SELECTOR",
            "DBT_FEATURE_BUILD_TEST_SELECTOR",
            DEFAULT_FEATURE_TEST_SELECTOR,
        ),
        default_selector=DEFAULT_FEATURE_TEST_SELECTOR,
    )
    threads = _get_feature_threads()
    summary = run_feature_smoke_tests(
        project_dir=_get_setting("DBT_PROJECT_DIR", "DBT_PROJECT_DIR", "dbt/bikeshare_dbt"),
        profiles_dir=_get_setting("DBT_PROFILES_DIR", "DBT_PROFILES_DIR", "dbt"),
        select_models=test_select,
        selector=None if test_select else test_selector,
        threads=threads,
        dbt_vars=_build_feature_test_vars(context),
    )
    print(f"AIRFLOW_TASK_METRIC run_dbt_feature_smoke_tests {summary}")


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(seconds=30),
}

start = pendulum.datetime(2026, 3, 1, tz="Europe/Paris")

with DAG(
    dag_id="dbt_feature_build_5min",
    start_date=start,
    schedule=DBT_FEATURE_BUILD_5MIN_SCHEDULE,
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["dbt", "feature_store", "5min", "contract_critical"],
) as dag:
    wait_for_hotpath = ExternalTaskSensor(
        task_id="wait_for_station_status_hotpath",
        external_dag_id="dbt_station_status_hotpath_5min",
        external_task_id=None,
        allowed_states=["success"],
        failed_states=["failed"],
        execution_date_fn=execution_date_fn_for_schedule(DBT_STATION_STATUS_HOTPATH_5MIN_SCHEDULE),
        check_existence=True,
        mode="reschedule",
        poke_interval=30,
        timeout=15 * 60,
        queue=_get_queue_name(),
    )
    build_and_test_features = PythonOperator(
        task_id="run_dbt_feature_build",
        python_callable=run_dbt_feature_build_task,
        queue=_get_queue_name(),
        pool=_get_pool_name(),
    )
    smoke_test_features = PythonOperator(
        task_id="run_dbt_feature_smoke_tests",
        python_callable=run_dbt_feature_smoke_tests_task,
        queue=_get_queue_name(),
        pool=_get_pool_name(),
    )

    wait_for_hotpath >> build_and_test_features >> smoke_test_features
