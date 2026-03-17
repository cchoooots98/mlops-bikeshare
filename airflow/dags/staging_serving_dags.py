import os
import sys

import pendulum

AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
if AIRFLOW_HOME not in sys.path:
    sys.path.append(AIRFLOW_HOME)

from serving_dag_factory import build_serving_dags


start = pendulum.datetime(2026, 3, 16, tz="Europe/Paris")

staging_prediction_dag, staging_quality_dag, staging_metrics_dag, staging_psi_dag = build_serving_dags(
    start=start,
    environment="staging",
    prediction_dag_id="staging_prediction_15min",
    quality_dag_id="staging_quality_backfill_15min",
    metrics_dag_id="staging_metrics_publish_hourly",
    psi_dag_id="staging_psi_publish_hourly",
)
