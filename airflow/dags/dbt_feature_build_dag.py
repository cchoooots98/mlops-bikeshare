import os
import sys
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
if AIRFLOW_HOME not in sys.path:
    sys.path.append(AIRFLOW_HOME)

from src.orchestration.dbt_tasks import (
    DEFAULT_FEATURE_BUILD_SELECT,
    DEFAULT_FEATURE_TEST_SELECT,
    parse_bool,
    parse_select_models,
    run_feature_model_build,
    run_feature_smoke_tests,
)


def _get_setting(var_key: str, env_key: str, default_value: str) -> str:
    return Variable.get(var_key, default_var=os.getenv(env_key, default_value))


def _get_pool_name() -> str:
    return _get_setting("DBT_FEATURE_POOL", "DBT_FEATURE_POOL", "dbt_feature_pool")


def _get_queue_name() -> str:
    return _get_setting("AIRFLOW_TIER1_QUEUE", "AIRFLOW_TIER1_QUEUE", "tier1")


def _build_feature_model_vars(context: dict) -> dict[str, object]:
    return {
        "feature_window_end_utc": context["data_interval_end"].astimezone(pendulum.UTC).isoformat(),
        "feature_rebuild_lookback_minutes": int(
            _get_setting("DBT_FEATURE_REBUILD_LOOKBACK_MINUTES", "DBT_FEATURE_REBUILD_LOOKBACK_MINUTES", "180"),
        ),
    }


def _build_feature_test_vars(context: dict) -> dict[str, object]:
    return {
        "test_window_end_utc": context["data_interval_end"].astimezone(pendulum.UTC).isoformat(),
        "test_window_lookback_hours": int(
            _get_setting("DBT_FEATURE_SMOKE_LOOKBACK_HOURS", "DBT_FEATURE_SMOKE_LOOKBACK_HOURS", "72"),
        ),
    }


def run_dbt_feature_build_task(**context):
    model_select = parse_select_models(
        _get_setting(
            "DBT_FEATURE_BUILD_SELECT",
            "DBT_FEATURE_BUILD_SELECT",
            " ".join(DEFAULT_FEATURE_BUILD_SELECT),
        ),
        default_models=DEFAULT_FEATURE_BUILD_SELECT,
    )
    threads = int(_get_setting("DBT_THREADS", "DBT_THREADS", "1"))
    summary = run_feature_model_build(
        project_dir=_get_setting("DBT_PROJECT_DIR", "DBT_PROJECT_DIR", "dbt/bikeshare_dbt"),
        profiles_dir=_get_setting("DBT_PROFILES_DIR", "DBT_PROFILES_DIR", "dbt"),
        select_models=model_select,
        selector=None,
        threads=threads,
        dbt_vars=_build_feature_model_vars(context),
    )
    print(f"AIRFLOW_TASK_METRIC run_dbt_feature_build {summary}")


def run_dbt_feature_smoke_tests_task(**context):
    if parse_bool(_get_setting("DBT_FEATURE_BUILD_SKIP_TESTS", "DBT_FEATURE_BUILD_SKIP_TESTS", "false")):
        summary = {"skipped": True}
        print(f"AIRFLOW_TASK_METRIC run_dbt_feature_smoke_tests {summary}")
        return
    test_select = parse_select_models(
        _get_setting(
            "DBT_FEATURE_BUILD_TEST_SELECT",
            "DBT_FEATURE_BUILD_TEST_SELECT",
            " ".join(DEFAULT_FEATURE_TEST_SELECT),
        ),
        default_models=DEFAULT_FEATURE_TEST_SELECT,
    )
    threads = int(_get_setting("DBT_THREADS", "DBT_THREADS", "1"))
    summary = run_feature_smoke_tests(
        project_dir=_get_setting("DBT_PROJECT_DIR", "DBT_PROJECT_DIR", "dbt/bikeshare_dbt"),
        profiles_dir=_get_setting("DBT_PROFILES_DIR", "DBT_PROFILES_DIR", "dbt"),
        select_models=test_select,
        selector=None,
        threads=threads,
        dbt_vars=_build_feature_test_vars(context),
    )
    print(f"AIRFLOW_TASK_METRIC run_dbt_feature_smoke_tests {summary}")


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

start = pendulum.datetime(2026, 3, 1, tz="Europe/Paris")

with DAG(
    dag_id="dbt_feature_build_5min",
    start_date=start,
    schedule="*/5 * * * *",
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
