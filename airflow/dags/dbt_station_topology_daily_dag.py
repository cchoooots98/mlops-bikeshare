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

from src.orchestration.dbt_tasks import DEFAULT_STATION_TOPOLOGY_SELECT, parse_select_models, run_model_build


def _get_setting(var_key: str, env_key: str, default_value: str) -> str:
    return Variable.get(var_key, default_var=os.getenv(env_key, default_value))


def _get_pool_name() -> str:
    return _get_setting("DBT_QUALITY_POOL", "DBT_QUALITY_POOL", "dbt_quality_pool")


def _get_queue_name() -> str:
    return _get_setting("AIRFLOW_TIER2_QUEUE", "AIRFLOW_TIER2_QUEUE", "tier2")


def run_dbt_station_topology_task():
    model_select = parse_select_models(
        _get_setting(
            "DBT_STATION_TOPOLOGY_SELECT",
            "DBT_STATION_TOPOLOGY_SELECT",
            " ".join(DEFAULT_STATION_TOPOLOGY_SELECT),
        ),
        default_models=DEFAULT_STATION_TOPOLOGY_SELECT,
    )
    summary = run_model_build(
        project_dir=_get_setting("DBT_PROJECT_DIR", "DBT_PROJECT_DIR", "dbt/bikeshare_dbt"),
        profiles_dir=_get_setting("DBT_PROFILES_DIR", "DBT_PROFILES_DIR", "dbt"),
        select_models=model_select,
        selector=None,
        threads=int(_get_setting("DBT_THREADS", "DBT_THREADS", "2")),
    )
    print(f"AIRFLOW_TASK_METRIC run_dbt_station_topology {summary}")


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

start = pendulum.datetime(2026, 3, 1, tz="Europe/Paris")

with DAG(
    dag_id="dbt_station_topology_daily",
    start_date=start,
    schedule="9 2 * * *",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["dbt", "station_topology", "daily", "contract_critical"],
) as dag:
    wait_for_station_information = ExternalTaskSensor(
        task_id="wait_for_station_information_ingest",
        external_dag_id="gbfs_station_information_daily",
        external_task_id=None,
        allowed_states=["success"],
        failed_states=["failed"],
        execution_delta=timedelta(minutes=7),
        check_existence=True,
        mode="reschedule",
        poke_interval=60,
        timeout=60 * 60,
        queue=_get_queue_name(),
    )
    run_station_topology = PythonOperator(
        task_id="run_dbt_station_topology",
        python_callable=run_dbt_station_topology_task,
        queue=_get_queue_name(),
        pool=_get_pool_name(),
    )

    wait_for_station_information >> run_station_topology
