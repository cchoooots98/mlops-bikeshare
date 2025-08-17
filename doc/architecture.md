# System Architecture and Data Flow

## Overview
The **mlops-bikeshare** project implements a minimal but production-grade MLOps loop to predict 30-minute bike/dock stockouts at bikeshare stations. It integrates **data ingestion, feature engineering, training, model registry, deployment, inference, monitoring, and visualization**.

## Components
- **Data Sources**
  - **GBFS API**: Station status and information feeds (sampled every 1–5 minutes).
  - **Meteostat API**: Historical and real-time weather (temperature, precipitation, wind, etc.).

- **Ingestion & Storage**
  - **Lambda (Ingest Functions)**: Periodic (5 min) ingestion with schema validation (Pandera / Great Expectations).
  - **Amazon S3**: Partitioned storage for `raw/`, `curated/`, `features/`, `inference/`, and `monitoring/`.
  - **AWS Glue & Athena**: Metadata cataloging and SQL queries.

- **Feature Engineering & Training**
  - **Batch Feature Jobs**: Aggregations over time windows, neighborhood summaries, temporal and weather features.
  - **SageMaker Training**: Training XGBoost/LightGBM models with MLflow or SageMaker Experiments for tracking.

- **Model Registry & Deployment**
  - **Option A**: SageMaker Model Registry and Pipelines (with conditional deployment).
  - **Option B**: MLflow Registry with `mlflow.sagemaker.deploy`.
  - **Endpoints**: SageMaker real-time inference endpoints (staging → production).

- **Inference**
  - **Lambda (Online Features)**: Joins current status + cached static info + weather.
  - **Handler**: Sends batches to the SageMaker endpoint.
  - **S3 Inference Logs**: Results stored for monitoring and evaluation.

- **Monitoring**
  - **SageMaker Model Monitor**: Data quality, drift detection, model quality baselines.
  - **CloudWatch**: System metrics (latency, errors, throughput) + custom KPIs.
  - **SNS / Slack / Email**: Notifications for alarms.

- **Visualization**
  - **Streamlit Dashboard**: Business view—map heatmaps, top-N risky stations, model/system health, data freshness.
  - **CloudWatch Dashboards / Grafana**: System-level observability.

## Data Flow (ASCII)
[GBFS API] [Meteostat]
\ /
[Lambda Ingest (5m)] -- Schema Validation --> [S3 Raw] -- Glue/Athena --
[Batch Features] -> [S3 Features]
|
v
[SageMaker Training] -- MLflow/SM Experiments
|
(A) SM Model Registry ---------| (B) MLflow Registry
|
[SageMaker Endpoint (staging → prod)]
^
[Lambda Inference (5–10m)] -- featurize_online ----------------------| [Model Monitor]
| |
v v
[S3 Inference + Monitoring] -- Athena --------------------------------------> [Streamlit Dashboard]
[CloudWatch/Grafana/Alarms]


## Account Boundaries
- All infrastructure provisioned under **AWS Account 387706002632**, region **ca-central-1**.
- IAM roles are least-privilege: GitHub OIDC deployer, SageMaker execution role, Lambda execution role.
- Separation between dev/staging/prod ensured through GitHub Actions workflows.

