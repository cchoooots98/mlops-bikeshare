# MLOps Bikeshare

This repository implements a dual-target, dbt-first bikeshare platform for Paris:
- local development and offline training on a reproducible Python 3.11 environment
- always-on data engineering on EC2 with Docker Compose
- AWS serving with SageMaker staging/prod endpoints, target-aware monitoring, and rollback

## Canonical Local Defaults
- Python environment: `.venv`
- MLflow tracking URI: `sqlite:///model_dir/mlflow.db`
- Local artifact root: `mlruns/`
- Postgres env vars: `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`
- Local deployment state:
  - `model_dir/deployments/bikes/local.json`
  - `model_dir/deployments/docks/local.json`
- Local package roots:
  - `model_dir/packages/bikes/`
  - `model_dir/packages/docks/`

Compatibility aliases such as `DW_*`, `RAW_S3_BUCKET`, and `WEATHER_CITY` are still read for one transition cycle, but they are no longer the formal contract.

## Quick Start
1. Create a local virtual environment.
2. Install locked runtime and dev dependencies into `.venv`.
3. Start Docker Compose for local Postgres and Airflow.
4. Build dbt features, then run offline training for bikes or docks.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
docker compose up -d
.\.venv\Scripts\python.exe -m pytest -q
```

Copy the template environment file before local work:

```powershell
Copy-Item .env.example .env
```

## Local Training Examples
Train bikes:

```powershell
.\.venv\Scripts\python.exe -m src.training.train `
  --city paris `
  --start "2026-03-01 00:00" `
  --end "2026-03-07 23:55" `
  --predict-bikes true `
  --model-type xgboost
```

Train docks:

```powershell
.\.venv\Scripts\python.exe -m src.training.train `
  --city paris `
  --start "2026-03-01 00:00" `
  --end "2026-03-07 23:55" `
  --predict-bikes false `
  --model-type xgboost
```

These commands write packages under target-specific roots and log runs to the local SQLite-backed MLflow store by default.

## Runtime Configuration
The formal runtime contract for inference and orchestration is:

```powershell
$env:PGHOST = "localhost"
$env:PGPORT = "15432"
$env:PGDATABASE = "velib_dw"
$env:PGUSER = "velib"
$env:PGPASSWORD = "velib"

$env:AWS_REGION = "eu-west-3"
$env:BUCKET = "bikeshare-paris-dev"
$env:CITY = "paris"
$env:TARGET_NAME = "bikes"
$env:SERVING_ENVIRONMENT = "local"
$env:DEPLOYMENT_STATE_PATH = "model_dir/deployments/bikes/local.json"
$env:MLFLOW_TRACKING_URI = "sqlite:///model_dir/mlflow.db"
```

## Repository Structure
- `src/config/`: runtime settings, target resolution, and naming helpers
- `src/training/`: offline training and evaluation
- `src/inference/`: online feature loading and prediction publishing
- `src/model_package/`: package manifests and deployment state records
- `src/monitoring/`: quality backfill, CloudWatch metrics, and monitoring helpers
- `src/serving/`: router request parsing and target resolution
- `pipelines/`: export, deploy, promote, and rollback entrypoints
- `infra/terraform/`: dev/prod envs and platform module
- `docs/`: execution guide, deployment guide, dashboard spec, and operator runbooks

## Formal Guides
Open these local files directly:
- [docs/project_documents.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/project_documents.md)
- [docs/execution_guide.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/execution_guide.md)
- [docs/deployment_guide.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/deployment_guide.md)
- [docs/runbook_prod.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/runbook_prod.md)
- [docs/dashboard_spec.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/dashboard_spec.md)
- [docs/plan_detail/current_state_to_enterprise_operator_manual.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/plan_detail/current_state_to_enterprise_operator_manual.md)

If you are actively building or rebuilding the project, start with the operator manual.
Use `docs/deployment_guide.md` as the command authority and `docs/evidence_capture_template.md` to save phase evidence.

## Validation
Use the virtual environment for all formal validation:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
docker compose ps
```

The repository should not be considered Phase 0 complete until those commands succeed from `.venv`.
