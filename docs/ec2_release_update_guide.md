# EC2 Release Update Guide

## Purpose
Use this guide after you have already validated a branch locally and want to refresh the EC2 runtime with the new code.

## Local Preconditions
Run these on your workstation before touching EC2:

```powershell
git status --short --branch
.\.venv\Scripts\python.exe -m pytest -q
docker compose -f docker-compose.yml -f docker-compose.local.yml ps
git rev-parse HEAD
git push origin <your-branch>
```

Use the exact branch or commit you just validated locally.

## EC2 Refresh Procedure
Run these on the EC2 host from the repository root:

```bash
git fetch origin
git checkout <your-branch>
git pull --ff-only origin <your-branch>
git rev-parse HEAD
docker compose up -d airflow-postgres dw-postgres mlflow-postgres mlflow
docker compose up -d redis
docker compose up airflow-init
docker compose up -d --build --force-recreate airflow-webserver airflow-scheduler airflow-worker-tier1 airflow-worker-tier2
docker compose ps
docker compose exec airflow-webserver airflow dags list-import-errors
docker compose exec airflow-webserver airflow dags list
```

Notes:
- `docker-compose.yml` is the formal EC2 baseline.
- Do not use `docker-compose.local.yml` on EC2.
- Airflow now mounts `./model_dir` to `/opt/airflow/model_dir`, so deployment-state and package paths resolve consistently inside the container.
- Airflow now runs with `CeleryExecutor` on a single EC2 host:
  - `redis` is the local broker
  - `airflow-worker-tier1` handles the prediction-critical path
  - `airflow-worker-tier2` handles quality, metrics, PSI, and heavier dbt observation work

## Mandatory Smoke Check
Before you unpause any staging DAG again, smoke-test both prediction tasks:

```bash
docker compose exec airflow-webserver airflow tasks test staging_prediction_15min predict_bikes <logical-date>
docker compose exec airflow-webserver airflow tasks test staging_prediction_15min predict_docks <logical-date>
```

Only continue if both tasks succeed and `airflow dags list-import-errors` is empty.

The serving DAG timing contract is:
- prediction runs every 15 minutes
- quality backfill runs 37 minutes after the prediction run it depends on
- that 37-minute offset includes the 30-minute label horizon plus a 7-minute buffer slot
- metrics waits 5 minutes after quality
- PSI runs independently from metrics and instead fails fast when feature freshness is missing

## After Smoke Passes
If you are in the staging observation window, unpause only:
- `staging_prediction_15min`
- `staging_quality_backfill_15min`
- `staging_metrics_publish_hourly`
- `staging_psi_publish_hourly`

Keep paused:
- `serving_prediction_15min`
- `serving_quality_backfill_15min`
- `serving_metrics_publish_hourly`
- `serving_psi_publish_hourly`

## Evidence To Keep
- deployed branch name
- deployed commit SHA
- `docker compose ps`
- `airflow dags list-import-errors`
- smoke-test outputs for `predict_bikes` and `predict_docks`
