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

from src.orchestration.dbt_tasks import parse_select_models, run_feature_build


def _get_setting(var_key: str, env_key: str, default_value: str) -> str:
    return Variable.get(var_key, default_var=os.getenv(env_key, default_value))


def run_dbt_feature_build_task():
    select_models = parse_select_models(
        _get_setting("DBT_FEATURE_BUILD_SELECT", "DBT_FEATURE_BUILD_SELECT", ""),
    )
    run_feature_build(
        project_dir=_get_setting("DBT_PROJECT_DIR", "DBT_PROJECT_DIR", "dbt/bikeshare_dbt"),
        profiles_dir=_get_setting("DBT_PROFILES_DIR", "DBT_PROFILES_DIR", "dbt"),
        select_models=select_models,
    )


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
    build_and_test_features = PythonOperator(
        task_id="run_dbt_feature_build",
        python_callable=run_dbt_feature_build_task,
    )

    wait_for_station_status>> build_and_test_features
