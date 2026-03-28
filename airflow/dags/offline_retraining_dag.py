import os
import sys
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
if AIRFLOW_HOME not in sys.path:
    sys.path.append(AIRFLOW_HOME)

from queue_defs import DAILY_SIDECAR_QUEUE
from runtime_utils import get_airflow_setting as _get_setting, get_dw_connection
from src.config import run_project_module
from schedule_defs import OFFLINE_MODEL_RETRAINING_DAILY_SCHEDULE


def _get_queue_name() -> str:
    return _get_setting("AIRFLOW_QUEUE_DAILY_SIDECAR", "AIRFLOW_QUEUE_DAILY_SIDECAR", DAILY_SIDECAR_QUEUE)


def _dag_conf(context: dict) -> dict:
    dag_run = context.get("dag_run")
    return dag_run.conf if dag_run and dag_run.conf else {}


def run_retraining_task(**context):
    conf = _dag_conf(context)
    dw_conn = get_dw_connection()
    args = [
        "--reason",
        str(conf.get("reason", "schedule")),
        "--city",
        str(conf.get("city", _get_setting("TRAIN_CITY", "TRAIN_CITY", "paris"))),
        "--predict-bikes",
        str(conf.get("predict_bikes", _get_setting("TRAIN_PREDICT_BIKES", "TRAIN_PREDICT_BIKES", "true"))),
        "--model-type",
        str(conf.get("model_type", _get_setting("TRAIN_MODEL_TYPE", "TRAIN_MODEL_TYPE", "xgboost"))),
        "--lookback-days",
        str(conf.get("lookback_days", _get_setting("TRAIN_LOOKBACK_DAYS", "TRAIN_LOOKBACK_DAYS", "30"))),
        "--pg-host",
        dw_conn.host,
        "--pg-port",
        str(dw_conn.port or 5432),
        "--pg-db",
        dw_conn.schema,
        "--pg-user",
        dw_conn.login,
        "--pg-password",
        dw_conn.password,
        "--pg-schema",
        _get_setting("TRAIN_PG_SCHEMA", "TRAIN_PG_SCHEMA", "analytics"),
        "--feature-table",
        _get_setting("TRAIN_FEATURE_TABLE", "TRAIN_FEATURE_TABLE", "feat_station_snapshot_5min"),
        "--experiment",
        _get_setting("TRAIN_EXPERIMENT", "TRAIN_EXPERIMENT", "bikeshare-offline-retrain"),
        "--dbt-project-dir",
        _get_setting("DBT_PROJECT_DIR", "DBT_PROJECT_DIR", "dbt/bikeshare_dbt"),
        "--dbt-profiles-dir",
        _get_setting("DBT_PROFILES_DIR", "DBT_PROFILES_DIR", "dbt"),
        "--summary-path",
        _get_setting("RETRAIN_SUMMARY_PATH", "RETRAIN_SUMMARY_PATH", "model_dir/candidates/retrain_summary.json"),
    ]
    return run_project_module(
        "src.orchestration.retrain",
        args=args,
        cwd=AIRFLOW_HOME,
        result_prefix="RETRAIN_RESULT_JSON::",
    )


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

start = pendulum.datetime(2026, 3, 11, tz="Europe/Paris")

with DAG(
    dag_id="offline_model_retraining_daily",
    start_date=start,
    schedule=OFFLINE_MODEL_RETRAINING_DAILY_SCHEDULE,
    catchup=False,
    default_args=default_args,
    tags=["training", "offline", "candidate"],
) as dag:
    PythonOperator(
        task_id="run_offline_retraining",
        python_callable=run_retraining_task,
        queue=_get_queue_name(),
    )
