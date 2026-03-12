import os
import sys
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.hooks.base import BaseHook
from airflow.models import Variable
from airflow.operators.python import PythonOperator

AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
if AIRFLOW_HOME not in sys.path:
    sys.path.append(AIRFLOW_HOME)

from src.orchestration.retrain import main as retrain_main


def _get_setting(var_key: str, env_key: str, default_value: str) -> str:
    return Variable.get(var_key, default_var=os.getenv(env_key, default_value))


def _dw_connection():
    conn_id = _get_setting("DW_CONN_ID", "DW_CONN_ID", "velib_dw")
    return BaseHook.get_connection(conn_id)


def _dag_conf(context: dict) -> dict:
    dag_run = context.get("dag_run")
    return dag_run.conf if dag_run and dag_run.conf else {}


def run_retraining_task(**context):
    conf = _dag_conf(context)
    dw_conn = _dw_connection()
    args = [
        "--reason",
        str(conf.get("reason", "schedule")),
        "--city",
        str(conf.get("city", _get_setting("TRAIN_CITY", "TRAIN_CITY", "paris"))),
        "--label",
        str(conf.get("label", _get_setting("TRAIN_LABEL", "TRAIN_LABEL", "y_stockout_bikes_30"))),
        "--model-type",
        str(conf.get("model_type", _get_setting("TRAIN_MODEL_TYPE", "TRAIN_MODEL_TYPE", "xgboost"))),
        "--lookback-days",
        str(conf.get("lookback_days", _get_setting("TRAIN_LOOKBACK_DAYS", "TRAIN_LOOKBACK_DAYS", "30"))),
        "--deploy-staging",
        str(conf.get("deploy_staging", _get_setting("TRAIN_DEPLOY_STAGING", "TRAIN_DEPLOY_STAGING", "false"))),
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
        "--manifest-path",
        _get_setting("RETRAIN_MANIFEST_PATH", "RETRAIN_MANIFEST_PATH", "model_dir/retrain_manifest.json"),
    ]
    retrain_main(args)


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

start = pendulum.datetime(2026, 3, 11, tz="Europe/Paris")

with DAG(
    dag_id="offline_model_retraining_daily",
    start_date=start,
    schedule="30 3 * * *",
    catchup=False,
    default_args=default_args,
    tags=["training", "offline", "candidate"],
) as dag:
    PythonOperator(task_id="run_offline_retraining", python_callable=run_retraining_task)
