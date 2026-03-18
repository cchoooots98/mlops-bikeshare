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
    DEFAULT_HOTPATH_BUILD_SELECT,
    DEFAULT_HOTPATH_TEST_SELECT,
    check_dim_station_staleness,
    check_dim_weather_staleness,
    parse_select_models,
    run_model_tests,
    run_model_build,
)


def _get_setting(var_key: str, env_key: str, default_value: str) -> str:
    return Variable.get(var_key, default_var=os.getenv(env_key, default_value))


def _get_pool_name() -> str:
    return _get_setting("DBT_HOTPATH_POOL", "DBT_HOTPATH_POOL", "dbt_hotpath_pool")


def _get_queue_name() -> str:
    return _get_setting("AIRFLOW_TIER1_QUEUE", "AIRFLOW_TIER1_QUEUE", "tier1")


def _get_city() -> str:
    return _get_setting("CITY", "CITY", "paris")


def _build_hotpath_model_vars(context: dict) -> dict[str, object]:
    return {
        "hotpath_window_end_utc": context["data_interval_end"].astimezone(pendulum.UTC).isoformat(),
        "station_status_rebuild_lookback_minutes": int(
            _get_setting("DBT_STATION_STATUS_REBUILD_LOOKBACK_MINUTES", "DBT_STATION_STATUS_REBUILD_LOOKBACK_MINUTES", "60"),
        ),
        "enrich_rebuild_lookback_minutes": int(
            _get_setting("DBT_ENRICH_REBUILD_LOOKBACK_MINUTES", "DBT_ENRICH_REBUILD_LOOKBACK_MINUTES", "120"),
        ),
    }


def _build_hotpath_test_vars(context: dict) -> dict[str, object]:
    return {
        "test_window_end_utc": context["data_interval_end"].astimezone(pendulum.UTC).isoformat(),
        "test_window_lookback_hours": int(
            _get_setting("DBT_HOTPATH_TEST_LOOKBACK_HOURS", "DBT_HOTPATH_TEST_LOOKBACK_HOURS", "2"),
        ),
    }


def check_dim_weather_staleness_task():
    summary = check_dim_weather_staleness(
        host=_get_setting("DBT_HOST", "DBT_HOST", "dw-postgres"),
        port=int(_get_setting("DBT_PORT", "DBT_PORT", "5432")),
        database=_get_setting("DBT_DATABASE", "DBT_DATABASE", "velib_dw"),
        user=_get_setting("DBT_USER", "DBT_USER", "velib"),
        password=_get_setting("DBT_PASSWORD", "DBT_PASSWORD", "velib"),
        schema=_get_setting("DBT_SCHEMA", "DBT_SCHEMA", "analytics"),
        city=_get_city(),
        stale_error_minutes=int(
            _get_setting("DBT_DIM_WEATHER_STALE_ERROR_MINUTES", "DBT_DIM_WEATHER_STALE_ERROR_MINUTES", "60"),
        ),
    )
    print(f"AIRFLOW_TASK_METRIC check_dim_weather_staleness {summary}")


def check_dim_station_staleness_task():
    summary = check_dim_station_staleness(
        host=_get_setting("DBT_HOST", "DBT_HOST", "dw-postgres"),
        port=int(_get_setting("DBT_PORT", "DBT_PORT", "5432")),
        database=_get_setting("DBT_DATABASE", "DBT_DATABASE", "velib_dw"),
        user=_get_setting("DBT_USER", "DBT_USER", "velib"),
        password=_get_setting("DBT_PASSWORD", "DBT_PASSWORD", "velib"),
        schema=_get_setting("DBT_SCHEMA", "DBT_SCHEMA", "analytics"),
        city=_get_city(),
        stale_warn_hours=int(
            _get_setting("DBT_DIM_STATION_STALE_WARN_HOURS", "DBT_DIM_STATION_STALE_WARN_HOURS", "73"),
        ),
    )
    print(f"AIRFLOW_TASK_METRIC check_dim_station_staleness {summary}")


def run_dbt_station_status_hotpath_task(**context):
    model_select = parse_select_models(
        _get_setting(
            "DBT_STATION_STATUS_HOTPATH_SELECT",
            "DBT_STATION_STATUS_HOTPATH_SELECT",
            " ".join(DEFAULT_HOTPATH_BUILD_SELECT),
        ),
        default_models=DEFAULT_HOTPATH_BUILD_SELECT,
    )
    summary = run_model_build(
        project_dir=_get_setting("DBT_PROJECT_DIR", "DBT_PROJECT_DIR", "dbt/bikeshare_dbt"),
        profiles_dir=_get_setting("DBT_PROFILES_DIR", "DBT_PROFILES_DIR", "dbt"),
        select_models=model_select,
        selector=None,
        threads=int(_get_setting("DBT_THREADS", "DBT_THREADS", "1")),
        dbt_vars=_build_hotpath_model_vars(context),
    )
    print(f"AIRFLOW_TASK_METRIC run_dbt_station_status_hotpath {summary}")


def run_dbt_station_status_hotpath_tests_task(**context):
    test_select = parse_select_models(
        _get_setting(
            "DBT_STATION_STATUS_HOTPATH_TEST_SELECT",
            "DBT_STATION_STATUS_HOTPATH_TEST_SELECT",
            " ".join(DEFAULT_HOTPATH_TEST_SELECT),
        ),
        default_models=DEFAULT_HOTPATH_TEST_SELECT,
    )
    summary = run_model_tests(
        project_dir=_get_setting("DBT_PROJECT_DIR", "DBT_PROJECT_DIR", "dbt/bikeshare_dbt"),
        profiles_dir=_get_setting("DBT_PROFILES_DIR", "DBT_PROFILES_DIR", "dbt"),
        select_models=test_select,
        selector=None,
        threads=int(_get_setting("DBT_THREADS", "DBT_THREADS", "1")),
        dbt_vars=_build_hotpath_test_vars(context),
        summary_label="DBT_HOTPATH_TEST_SUMMARY",
        indirect_selection="cautious",
    )
    print(f"AIRFLOW_TASK_METRIC run_dbt_station_status_hotpath_tests {summary}")



default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

start = pendulum.datetime(2026, 3, 1, tz="Europe/Paris")

with DAG(
    dag_id="dbt_station_status_hotpath_5min",
    start_date=start,
    schedule="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["dbt", "hotpath", "5min", "contract_critical"],
) as dag:
    wait_for_station_status = ExternalTaskSensor(
        task_id="wait_for_station_status_ingest",
        external_dag_id="gbfs_station_status_5min",
        external_task_id=None,
        allowed_states=["success"],
        failed_states=["failed"],
        check_existence=True,
        mode="reschedule",
        poke_interval=30,
        timeout=15 * 60,
        queue=_get_queue_name(),
    )
    dim_station_staleness_warn = PythonOperator(
        task_id="check_dim_station_staleness",
        python_callable=check_dim_station_staleness_task,
        queue=_get_queue_name(),
    )
    dim_weather_staleness_gate = PythonOperator(
        task_id="check_dim_weather_staleness",
        python_callable=check_dim_weather_staleness_task,
        queue=_get_queue_name(),
    )
    run_station_status_hotpath = PythonOperator(
        task_id="run_dbt_station_status_hotpath",
        python_callable=run_dbt_station_status_hotpath_task,
        queue=_get_queue_name(),
        pool=_get_pool_name(),
    )
    test_station_status_hotpath = PythonOperator(
        task_id="run_dbt_station_status_hotpath_tests",
        python_callable=run_dbt_station_status_hotpath_tests_task,
        queue=_get_queue_name(),
        pool=_get_pool_name(),
    )

    wait_for_station_status >> dim_station_staleness_warn >> dim_weather_staleness_gate >> run_station_status_hotpath >> test_station_status_hotpath
