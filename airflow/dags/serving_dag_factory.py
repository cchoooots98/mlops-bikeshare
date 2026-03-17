import os
import subprocess
import sys
from datetime import timedelta

from airflow import DAG
from airflow.hooks.base import BaseHook
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
if AIRFLOW_HOME not in sys.path:
    sys.path.append(AIRFLOW_HOME)

from src.config.naming import deployment_state_path, endpoint_name


TARGETS = ("bikes", "docks")
DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _get_setting(var_key: str, env_key: str, default_value: str) -> str:
    return Variable.get(var_key, default_var=os.getenv(env_key, default_value))


def _dw_connection():
    conn_id = _get_setting("DW_CONN_ID", "DW_CONN_ID", "velib_dw")
    return BaseHook.get_connection(conn_id)


def _base_runtime_env(*, target_name: str, environment: str) -> dict[str, str]:
    connection = _dw_connection()
    return {
        "PGHOST": connection.host,
        "PGPORT": str(connection.port or 5432),
        "PGDATABASE": connection.schema,
        "PGUSER": connection.login,
        "PGPASSWORD": connection.password or "",
        "AWS_REGION": _get_setting("AWS_REGION", "AWS_REGION", "eu-west-3"),
        "CITY": _get_setting("CITY", "CITY", "paris"),
        "BUCKET": _get_setting("BUCKET", "BUCKET", ""),
        "CW_NS": _get_setting("CW_NS", "CW_NS", "Bikeshare/Model"),
        "TARGET_NAME": target_name,
        "SERVING_ENVIRONMENT": environment,
        "DEPLOYMENT_STATE_PATH": _get_setting(
            f"DEPLOYMENT_STATE_{target_name.upper()}_{environment.upper()}_PATH",
            f"DEPLOYMENT_STATE_{target_name.upper()}_{environment.upper()}_PATH",
            str(deployment_state_path(target_name=target_name, environment=environment)),
        ),
        "SM_ENDPOINT": _get_setting(
            f"SM_ENDPOINT_{target_name.upper()}_{environment.upper()}",
            f"SM_ENDPOINT_{target_name.upper()}_{environment.upper()}",
            endpoint_name(target_name=target_name, environment=environment),
        ),
    }


def _run_module(module_name: str, *, args: list[str] | None = None, extra_env: dict[str, str] | None = None) -> None:
    env = dict(os.environ)
    env.update(extra_env or {})
    subprocess.run(
        [sys.executable, "-m", module_name, *(args or [])],
        check=True,
        cwd=AIRFLOW_HOME,
        env=env,
    )


def run_prediction_task(*, target_name: str, environment: str) -> None:
    _run_module(
        "src.inference.predictor",
        extra_env=_base_runtime_env(target_name=target_name, environment=environment),
    )


def run_quality_backfill_task(*, target_name: str, environment: str) -> None:
    _run_module(
        "src.monitoring.quality_backfill",
        extra_env=_base_runtime_env(target_name=target_name, environment=environment),
    )


def run_metrics_publish_task(*, target_name: str, environment: str) -> None:
    env = _base_runtime_env(target_name=target_name, environment=environment)
    _run_module(
        "src.monitoring.metrics.publish_custom_metrics",
        args=[
            "--bucket",
            env["BUCKET"],
            "--quality-prefix",
            "AUTO",
            "--endpoint",
            env["SM_ENDPOINT"],
            "--city-dimension",
            env["CITY"],
            "--target-name",
            target_name,
            "--environment",
            environment,
        ],
        extra_env=env,
    )


def run_psi_publish_task(*, target_name: str, environment: str) -> None:
    env = _base_runtime_env(target_name=target_name, environment=environment)
    _run_module(
        "src.monitoring.metrics.publish_psi",
        args=[
            "--city",
            env["CITY"],
            "--endpoint",
            env["SM_ENDPOINT"],
            "--target-name",
            target_name,
            "--environment",
            environment,
            "--pg-host",
            env["PGHOST"],
            "--pg-port",
            env["PGPORT"],
            "--pg-db",
            env["PGDATABASE"],
            "--pg-user",
            env["PGUSER"],
            "--pg-password",
            env["PGPASSWORD"],
        ],
        extra_env=env,
    )


def build_serving_dags(
    *,
    start,
    environment: str,
    prediction_dag_id: str,
    quality_dag_id: str,
    metrics_dag_id: str,
    psi_dag_id: str,
) -> tuple[DAG, DAG, DAG, DAG]:
    prediction_tags = ["serving", "prediction", environment]
    quality_tags = ["serving", "quality", environment]
    metrics_tags = ["serving", "metrics", environment]
    psi_tags = ["serving", "drift", environment]

    with DAG(
        dag_id=prediction_dag_id,
        start_date=start,
        schedule="*/15 * * * *",
        catchup=False,
        max_active_runs=1,
        default_args=DEFAULT_ARGS,
        tags=prediction_tags,
    ) as prediction_dag:
        for target in TARGETS:
            PythonOperator(
                task_id=f"predict_{target}",
                python_callable=run_prediction_task,
                op_kwargs={"target_name": target, "environment": environment},
            )

    with DAG(
        dag_id=quality_dag_id,
        start_date=start,
        schedule="7,22,37,52 * * * *",
        catchup=False,
        max_active_runs=1,
        default_args=DEFAULT_ARGS,
        tags=quality_tags,
    ) as quality_dag:
        wait_for_prediction = ExternalTaskSensor(
            task_id="wait_for_prediction_dag",
            external_dag_id=prediction_dag_id,
            external_task_id=None,
            allowed_states=["success"],
            failed_states=["failed"],
            check_existence=True,
            mode="reschedule",
            poke_interval=60,
            timeout=15 * 60,
        )
        for target in TARGETS:
            task = PythonOperator(
                task_id=f"backfill_quality_{target}",
                python_callable=run_quality_backfill_task,
                op_kwargs={"target_name": target, "environment": environment},
            )
            wait_for_prediction >> task

    with DAG(
        dag_id=metrics_dag_id,
        start_date=start,
        schedule="12 * * * *",
        catchup=False,
        max_active_runs=1,
        default_args=DEFAULT_ARGS,
        tags=metrics_tags,
    ) as metrics_dag:
        wait_for_quality = ExternalTaskSensor(
            task_id="wait_for_quality_dag",
            external_dag_id=quality_dag_id,
            external_task_id=None,
            allowed_states=["success"],
            failed_states=["failed"],
            check_existence=True,
            mode="reschedule",
            poke_interval=60,
            timeout=20 * 60,
        )
        for target in TARGETS:
            task = PythonOperator(
                task_id=f"publish_metrics_{target}",
                python_callable=run_metrics_publish_task,
                op_kwargs={"target_name": target, "environment": environment},
            )
            wait_for_quality >> task

    with DAG(
        dag_id=psi_dag_id,
        start_date=start,
        schedule="18 * * * *",
        catchup=False,
        max_active_runs=1,
        default_args=DEFAULT_ARGS,
        tags=psi_tags,
    ) as psi_dag:
        wait_for_metrics = ExternalTaskSensor(
            task_id="wait_for_metrics_dag",
            external_dag_id=metrics_dag_id,
            external_task_id=None,
            allowed_states=["success"],
            failed_states=["failed"],
            check_existence=True,
            mode="reschedule",
            poke_interval=60,
            timeout=20 * 60,
        )
        for target in TARGETS:
            task = PythonOperator(
                task_id=f"publish_psi_{target}",
                python_callable=run_psi_publish_task,
                op_kwargs={"target_name": target, "environment": environment},
            )
            wait_for_metrics >> task

    return prediction_dag, quality_dag, metrics_dag, psi_dag
