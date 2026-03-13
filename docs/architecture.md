# System Architecture

## Overview
The platform is organized around three operating layers:
- data production: Python ingestion plus dbt models in Postgres
- model production: offline training, MLflow logging, and local model packages
- model activation: deployment-state files, staging/prod promotion, and rollback

The formal architecture is hybrid:
- local development for coding, tests, and offline model iteration
- EC2 for always-on data engineering and dashboard hosting
- AWS for serving, alerting, promotion, and rollback

## Data Flow
1. Python ingestion writes raw payloads and `public.stg_*`.
2. dbt builds curated and feature tables in `analytics.*`.
3. Offline training reads `analytics.feat_station_snapshot_5min`.
4. Online and batch inference read `analytics.feat_station_snapshot_latest`.
5. Training emits a package directory under `model_dir/packages/<target>/...`.
6. Activation writes a deployment-state record under `model_dir/deployments/<target>/<environment>.json`.

## Targets And Isolation
The platform supports two prediction targets on a shared code path:
- `bikes`
- `docks`

The following resources must remain target-specific:
- package roots
- deployment state
- S3 inference and monitoring partitions
- CloudWatch metric dimensions
- SageMaker endpoint names
- dashboard labels and queries

## Model Package
Each model package has this fixed structure:

```text
<package_dir>/
  package_manifest.json
  model/
  artifacts/
```

`package_manifest.json` is the static source of truth for:
- target definition
- threshold
- feature contract version
- feature order
- model identity
- registry metadata

## Deployment State
Deployment-state JSON is the dynamic source of truth for:
- active environment
- active package directory
- active registered model/version
- last activation timestamp
- active endpoint name when deployed

The formal layout is:

```text
model_dir/deployments/bikes/local.json
model_dir/deployments/bikes/staging.json
model_dir/deployments/bikes/production.json
model_dir/deployments/docks/local.json
model_dir/deployments/docks/staging.json
model_dir/deployments/docks/production.json
```

Single-file deployment state is legacy-only and must not be used for formal dual-target workflows.

## Runtime Contract
Formal runtime settings:
- Postgres: `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`
- Scope: `CITY`, `BUCKET`, `TARGET_NAME`, `SERVING_ENVIRONMENT`
- MLflow local default: `MLFLOW_TRACKING_URI=sqlite:///model_dir/mlflow.db`

Compatibility aliases such as `DW_*`, `RAW_S3_BUCKET`, and `WEATHER_CITY` are tolerated temporarily but should not appear in new documentation or automation.

## Operating Split
Local workstation:
- code changes
- unit and integration tests
- offline training and model debug

EC2:
- Docker Compose
- Airflow scheduler/webserver
- Postgres
- dbt jobs
- dashboard service

AWS:
- ECR
- S3
- IAM
- SageMaker staging/prod endpoints
- CloudWatch alarms and dashboards
- SNS notifications
- router lambda
- promote and rollback automation
