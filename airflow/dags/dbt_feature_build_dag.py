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
    check_weather_semantic_freshness,
    parse_bool,
    parse_selector,
    run_feature_model_build,
    run_feature_smoke_tests,
)


def _get_setting(var_key: str, env_key: str, default_value: str) -> str:
    return Variable.get(var_key, default_var=os.getenv(env_key, default_value))


def _get_pool_name() -> str:
    return _get_setting("DBT_AIRFLOW_POOL", "DBT_AIRFLOW_POOL", "dbt_warehouse_serial")


def _get_city() -> str:
    return _get_setting("WEATHER_CITY", "WEATHER_CITY", _get_setting("GBFS_CITY", "GBFS_CITY", "paris"))


def check_weather_semantic_freshness_task():
    summary = check_weather_semantic_freshness(
        host=_get_setting("DBT_HOST", "DBT_HOST", "dw-postgres"),
        port=int(_get_setting("DBT_PORT", "DBT_PORT", "5432")),
        database=_get_setting("DBT_DATABASE", "DBT_DATABASE", "velib_dw"),
        user=_get_setting("DBT_USER", "DBT_USER", "velib"),
        password=_get_setting("DBT_PASSWORD", "DBT_PASSWORD", "velib"),
        city=_get_city(),
        current_warn_minutes=int(
            _get_setting("DBT_WEATHER_CURRENT_WARN_MINUTES", "DBT_WEATHER_CURRENT_WARN_MINUTES", "30"),
        ),
        current_error_minutes=int(
            _get_setting("DBT_WEATHER_CURRENT_ERROR_MINUTES", "DBT_WEATHER_CURRENT_ERROR_MINUTES", "120"),
        ),
        hourly_warn_minutes=int(
            _get_setting("DBT_WEATHER_HOURLY_WARN_MINUTES", "DBT_WEATHER_HOURLY_WARN_MINUTES", "30"),
        ),
        hourly_error_minutes=int(
            _get_setting("DBT_WEATHER_HOURLY_ERROR_MINUTES", "DBT_WEATHER_HOURLY_ERROR_MINUTES", "120"),
        ),
        min_forecast_coverage_minutes=int(
            _get_setting(
                "DBT_WEATHER_MIN_FORECAST_COVERAGE_MINUTES",
                "DBT_WEATHER_MIN_FORECAST_COVERAGE_MINUTES",
                "30",
            ),
        ),
    )
    print(f"AIRFLOW_TASK_METRIC check_weather_semantic_freshness {summary}")


def run_dbt_feature_build_task():
    selector = parse_selector(
        _get_setting("DBT_FEATURE_BUILD_SELECTOR", "DBT_FEATURE_BUILD_SELECTOR", ""),
        "hf_feature_build_models",
    )
    threads = int(_get_setting("DBT_THREADS", "DBT_THREADS", "1"))
    summary = run_feature_model_build(
        project_dir=_get_setting("DBT_PROJECT_DIR", "DBT_PROJECT_DIR", "dbt/bikeshare_dbt"),
        profiles_dir=_get_setting("DBT_PROFILES_DIR", "DBT_PROFILES_DIR", "dbt"),
        selector=selector,
        threads=threads,
    )
    print(f"AIRFLOW_TASK_METRIC run_dbt_feature_build {summary}")


def run_dbt_feature_smoke_tests_task():
    if parse_bool(_get_setting("DBT_FEATURE_BUILD_SKIP_TESTS", "DBT_FEATURE_BUILD_SKIP_TESTS", "false")):
        summary = {"skipped": True}
        print(f"AIRFLOW_TASK_METRIC run_dbt_feature_smoke_tests {summary}")
        return
    selector = parse_selector(
        _get_setting("DBT_FEATURE_BUILD_TEST_SELECTOR", "DBT_FEATURE_BUILD_TEST_SELECTOR", ""),
        "hf_smoke_tests",
    )
    threads = int(_get_setting("DBT_THREADS", "DBT_THREADS", "1"))
    summary = run_feature_smoke_tests(
        project_dir=_get_setting("DBT_PROJECT_DIR", "DBT_PROJECT_DIR", "dbt/bikeshare_dbt"),
        profiles_dir=_get_setting("DBT_PROFILES_DIR", "DBT_PROFILES_DIR", "dbt"),
        selector=selector,
        threads=threads,
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
    wait_for_station_status = ExternalTaskSensor(
        task_id="wait_for_station_status_ingest",
        external_dag_id="gbfs_station_status_5min",
        external_task_id=None,
        allowed_states=["success"],
        failed_states=["failed"],
        execution_delta=timedelta(minutes=5),
        check_existence=True,
        mode="reschedule",
        poke_interval=30,
        timeout=15 * 60,
    )
    weather_semantic_gate = PythonOperator(
        task_id="check_weather_semantic_freshness",
        python_callable=check_weather_semantic_freshness_task,
    )
    build_and_test_features = PythonOperator(
        task_id="run_dbt_feature_build",
        python_callable=run_dbt_feature_build_task,
        pool=_get_pool_name(),
    )
    smoke_test_features = PythonOperator(
        task_id="run_dbt_feature_smoke_tests",
        python_callable=run_dbt_feature_smoke_tests_task,
        pool=_get_pool_name(),
    )

    wait_for_station_status >> weather_semantic_gate >> build_and_test_features >> smoke_test_features
