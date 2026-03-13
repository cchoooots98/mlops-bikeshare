import os
import sys
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
if AIRFLOW_HOME not in sys.path:
    sys.path.append(AIRFLOW_HOME)

from src.orchestration.dbt_tasks import (
    DEFAULT_QUALITY_TEST_SELECT,
    DEFAULT_SOURCE_FRESHNESS_SELECT,
    parse_select_models,
    run_dbt_source_freshness,
    run_quality_tests,
)


def _get_setting(var_key: str, env_key: str, default_value: str) -> str:
    return Variable.get(var_key, default_var=os.getenv(env_key, default_value))


def _get_pool_name() -> str:
    return _get_setting("DBT_AIRFLOW_POOL", "DBT_AIRFLOW_POOL", "dbt_warehouse_serial")


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
        threads=int(_get_setting("DBT_THREADS", "DBT_THREADS", "1")),
    )
    print(f"AIRFLOW_TASK_METRIC run_dbt_source_freshness {summary}")


def run_dbt_quality_tests_task(**context):
    test_select = parse_select_models(
        _get_setting(
            "DBT_QUALITY_TEST_SELECT",
            "DBT_QUALITY_TEST_SELECT",
            " ".join(DEFAULT_QUALITY_TEST_SELECT),
        ),
        default_models=DEFAULT_QUALITY_TEST_SELECT,
    )
    summary = run_quality_tests(
        project_dir=_get_setting("DBT_PROJECT_DIR", "DBT_PROJECT_DIR", "dbt/bikeshare_dbt"),
        profiles_dir=_get_setting("DBT_PROFILES_DIR", "DBT_PROFILES_DIR", "dbt"),
        select_models=test_select,
        selector=None,
        threads=int(_get_setting("DBT_THREADS", "DBT_THREADS", "1")),
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
    schedule="17 * * * *",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["dbt", "quality", "hourly", "contract_critical"],
) as dag:
    source_freshness = PythonOperator(
        task_id="run_dbt_source_freshness",
        python_callable=run_dbt_source_freshness_task,
        pool=_get_pool_name(),
    )
    quality_tests = PythonOperator(
        task_id="run_dbt_quality_tests",
        python_callable=run_dbt_quality_tests_task,
        pool=_get_pool_name(),
    )

    source_freshness >> quality_tests
