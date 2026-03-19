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
    DEFAULT_DEEP_QUALITY_TEST_SELECTOR,
    DEFAULT_FEATURE_BUILD_SELECT,
    parse_selector,
    parse_select_models,
    run_feature_model_build,
    run_quality_tests,
)


def _get_setting(var_key: str, env_key: str, default_value: str) -> str:
    return Variable.get(var_key, default_var=os.getenv(env_key, default_value))


def _get_pool_name() -> str:
    return _get_setting("DBT_QUALITY_POOL", "DBT_QUALITY_POOL", "dbt_quality_pool")


def _get_queue_name() -> str:
    return _get_setting("AIRFLOW_TIER2_QUEUE", "AIRFLOW_TIER2_QUEUE", "tier2")


def _build_feature_reconcile_vars(context: dict) -> dict[str, object]:
    lookback_days = int(
        _get_setting(
            "DBT_DIAGNOSTIC_FEATURE_REBUILD_LOOKBACK_DAYS",
            "DBT_DIAGNOSTIC_FEATURE_REBUILD_LOOKBACK_DAYS",
            "5",
        ),
    )
    return {
        "feature_window_end_utc": context["data_interval_end"].astimezone(pendulum.UTC).isoformat(),
        "feature_rebuild_lookback_minutes": lookback_days * 24 * 60,
    }


def run_dbt_feature_reconcile_5d_task(**context):
    model_select = parse_select_models(
        _get_setting(
            "DBT_DIAGNOSTIC_FEATURE_BUILD_SELECT",
            "DBT_DIAGNOSTIC_FEATURE_BUILD_SELECT",
            " ".join(DEFAULT_FEATURE_BUILD_SELECT),
        ),
        default_models=DEFAULT_FEATURE_BUILD_SELECT,
    )
    summary = run_feature_model_build(
        project_dir=_get_setting("DBT_PROJECT_DIR", "DBT_PROJECT_DIR", "dbt/bikeshare_dbt"),
        profiles_dir=_get_setting("DBT_PROFILES_DIR", "DBT_PROFILES_DIR", "dbt"),
        select_models=model_select,
        selector=None,
        threads=int(_get_setting("DBT_THREADS", "DBT_THREADS", "2")),
        dbt_vars=_build_feature_reconcile_vars(context),
    )
    print(f"AIRFLOW_TASK_METRIC run_dbt_feature_reconcile_5d {summary}")


def run_dbt_deep_quality_tests_task():
    raw_test_select = _get_setting(
        "DBT_DEEP_QUALITY_TEST_SELECT",
        "DBT_DEEP_QUALITY_TEST_SELECT",
        "",
    )
    test_select = parse_select_models(raw_test_select, default_models=[]) if raw_test_select.strip() else None
    test_selector = parse_selector(
        _get_setting(
            "DBT_DEEP_QUALITY_TEST_SELECTOR",
            "DBT_DEEP_QUALITY_TEST_SELECTOR",
            DEFAULT_DEEP_QUALITY_TEST_SELECTOR,
        ),
        default_selector=DEFAULT_DEEP_QUALITY_TEST_SELECTOR,
    )
    summary = run_quality_tests(
        project_dir=_get_setting("DBT_PROJECT_DIR", "DBT_PROJECT_DIR", "dbt/bikeshare_dbt"),
        profiles_dir=_get_setting("DBT_PROFILES_DIR", "DBT_PROFILES_DIR", "dbt"),
        select_models=test_select,
        selector=None if test_select else test_selector,
        threads=int(_get_setting("DBT_THREADS", "DBT_THREADS", "2")),
    )
    print(f"AIRFLOW_TASK_METRIC run_dbt_deep_quality_tests {summary}")


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

start = pendulum.datetime(2026, 3, 1, tz="Europe/Paris")

with DAG(
    dag_id="dbt_diagnostic_daily",
    start_date=start,
    schedule="47 2 * * *",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["dbt", "quality", "diagnostic", "daily"],
) as dag:
    feature_reconcile_5d = PythonOperator(
        task_id="run_dbt_feature_reconcile_5d",
        python_callable=run_dbt_feature_reconcile_5d_task,
        queue=_get_queue_name(),
        pool=_get_pool_name(),
    )
    deep_quality_tests = PythonOperator(
        task_id="run_dbt_deep_quality_tests",
        python_callable=run_dbt_deep_quality_tests_task,
        queue=_get_queue_name(),
        pool=_get_pool_name(),
    )

    feature_reconcile_5d >> deep_quality_tests
