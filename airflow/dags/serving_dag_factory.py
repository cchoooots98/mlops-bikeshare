import os
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

from src.config import run_project_module
from src.config.naming import deployment_state_path, endpoint_name


TARGETS = ("bikes", "docks")
PREDICTION_CADENCE_MINUTES = 15
QUALITY_LABEL_MATURITY_MINUTES = 30
QUALITY_START_LAG_MINUTES = 7
QUALITY_TO_PREDICTION_DELTA = timedelta(minutes=QUALITY_LABEL_MATURITY_MINUTES + QUALITY_START_LAG_MINUTES)
METRICS_TO_QUALITY_DELTA = timedelta(minutes=5)
TIER1_QUEUE = "tier1"
TIER2_QUEUE = "tier2"
SERVING_PREDICTION_POOL = "serving_prediction_pool"
SERVING_OBSERVABILITY_POOL = "serving_observability_pool"
DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
PREDICTION_SCHEDULE = "1-59/15 * * * *"
QUALITY_SCHEDULE = "8-59/15 * * * *"
METRICS_SCHEDULE = "43 * * * *"
PSI_SCHEDULE = "3 * * * *"


def _get_setting(var_key: str, env_key: str, default_value: str) -> str:
    return Variable.get(var_key, default_var=os.getenv(env_key, default_value))


def _dw_connection():
    conn_id = _get_setting("DW_CONN_ID", "DW_CONN_ID", "velib_dw")
    return BaseHook.get_connection(conn_id)


def _tier1_queue() -> str:
    return _get_setting("AIRFLOW_TIER1_QUEUE", "AIRFLOW_TIER1_QUEUE", TIER1_QUEUE)


def _tier2_queue() -> str:
    return _get_setting("AIRFLOW_TIER2_QUEUE", "AIRFLOW_TIER2_QUEUE", TIER2_QUEUE)


def _serving_prediction_pool() -> str:
    return _get_setting("SERVING_PREDICTION_POOL", "SERVING_PREDICTION_POOL", SERVING_PREDICTION_POOL)


def _serving_observability_pool() -> str:
    return _get_setting("SERVING_OBSERVABILITY_POOL", "SERVING_OBSERVABILITY_POOL", SERVING_OBSERVABILITY_POOL)


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
    run_project_module(
        module_name,
        args=args,
        extra_env=extra_env,
        cwd=AIRFLOW_HOME,
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
            "--aggregator",
            _get_setting("PSI_AGGREGATOR", "PSI_AGGREGATOR", "trimmed_mean"),
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
        schedule=PREDICTION_SCHEDULE,
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
                queue=_tier1_queue(),
                pool=_serving_prediction_pool(),
            )

    with DAG(
        dag_id=quality_dag_id,
        start_date=start,
        schedule=QUALITY_SCHEDULE,
        catchup=False,
        max_active_runs=1,
        default_args=DEFAULT_ARGS,
        tags=quality_tags,
    ) as quality_dag:
        for target in TARGETS:
            wait_for_prediction = ExternalTaskSensor(
                task_id=f"wait_for_predict_{target}",
                external_dag_id=prediction_dag_id,
                external_task_id=f"predict_{target}",
                allowed_states=["success"],
                failed_states=["failed", "upstream_failed"],
                check_existence=True,
                execution_delta=QUALITY_TO_PREDICTION_DELTA,
                mode="reschedule",
                poke_interval=60,
                timeout=15 * 60,
                queue=_tier2_queue(),
            )
            task = PythonOperator(
                task_id=f"backfill_quality_{target}",
                python_callable=run_quality_backfill_task,
                op_kwargs={"target_name": target, "environment": environment},
                queue=_tier2_queue(),
                pool=_serving_observability_pool(),
            )
            wait_for_prediction >> task

    with DAG(
        dag_id=metrics_dag_id,
        start_date=start,
        schedule=METRICS_SCHEDULE,
        catchup=False,
        max_active_runs=1,
        default_args=DEFAULT_ARGS,
        tags=metrics_tags,
    ) as metrics_dag:
        for target in TARGETS:
            wait_for_quality = ExternalTaskSensor(
                task_id=f"wait_for_backfill_quality_{target}",
                external_dag_id=quality_dag_id,
                external_task_id=f"backfill_quality_{target}",
                allowed_states=["success"],
                failed_states=["failed", "upstream_failed"],
                check_existence=True,
                execution_delta=METRICS_TO_QUALITY_DELTA,
                mode="reschedule",
                poke_interval=60,
                timeout=20 * 60,
                queue=_tier2_queue(),
            )
            task = PythonOperator(
                task_id=f"publish_metrics_{target}",
                python_callable=run_metrics_publish_task,
                op_kwargs={"target_name": target, "environment": environment},
                queue=_tier2_queue(),
                pool=_serving_observability_pool(),
            )
            wait_for_quality >> task

    with DAG(
        dag_id=psi_dag_id,
        start_date=start,
        schedule=PSI_SCHEDULE,
        catchup=False,
        max_active_runs=1,
        default_args=DEFAULT_ARGS,
        tags=psi_tags,
    ) as psi_dag:
        for target in TARGETS:
            PythonOperator(
                task_id=f"publish_psi_{target}",
                python_callable=run_psi_publish_task,
                op_kwargs={"target_name": target, "environment": environment},
                queue=_tier2_queue(),
                pool=_serving_observability_pool(),
            )

    return prediction_dag, quality_dag, metrics_dag, psi_dag
