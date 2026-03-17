import os
import sys

import pendulum

AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
if AIRFLOW_HOME not in sys.path:
    sys.path.append(AIRFLOW_HOME)

from serving_dag_factory import build_serving_dags


start = pendulum.datetime(2026, 3, 16, tz="Europe/Paris")

prediction_dag, quality_dag, metrics_dag, psi_dag = build_serving_dags(
    start=start,
    environment="production",
    prediction_dag_id="serving_prediction_15min",
    quality_dag_id="serving_quality_backfill_15min",
    metrics_dag_id="serving_metrics_publish_hourly",
    psi_dag_id="serving_psi_publish_hourly",
)
