# MLOps Bikeshare

This repository implements a dbt-first bikeshare platform with local development, offline training, local package-based inference, and AWS-ready serving scripts.

## Current Architecture
- Python ingestion owns raw landing and `public.stg_*`.
- dbt owns curated models and feature tables in Postgres.
- Offline training reads `analytics.feat_station_snapshot_5min`.
- Online/local batch prediction reads `analytics.feat_station_snapshot_latest`.
- Model metadata is no longer read from ad hoc retrain summaries. The source of truth is:
  - static: `model package/package_manifest.json`
  - dynamic: `deployment state JSON`

## Local Model Lifecycle
1. Build dbt feature tables.
2. Train with `python -m src.training.train ...`.
3. Produce a local model package under `model_dir/packages/...`.
4. Register the run as a candidate with `python -m src.orchestration.retrain ...`.
5. Activate a package for local inference through a deployment-state file.
6. Run inference against the active package contract.

## Key Paths
- Training package root: `model_dir/packages/`
- Local deployment state: `model_dir/deployments/local.json`
- Candidate retrain summary: `model_dir/candidates/retrain_summary.json`

## Runtime Configuration
Important runtime settings for inference and local orchestration:

```powershell
$env:PGHOST = "localhost"
$env:PGPORT = "5432"
$env:PGDATABASE = "velib_dw"
$env:PGUSER = "velib"
$env:PGPASSWORD = "velib"

$env:SM_ENDPOINT = "bikeshare-staging"
$env:DEPLOYMENT_STATE_PATH = "model_dir/deployments/local.json"
# Optional override if you want to bypass deployment state:
$env:MODEL_PACKAGE_DIR = ""
```

Deprecated compatibility aliases still map to `DEPLOYMENT_STATE_PATH` for one transition cycle:
- `MODEL_METADATA_PATH`
- `RETRAIN_MANIFEST_PATH`

## AWS Direction
AWS remains the serving target:
- SageMaker staging/prod endpoints
- CloudWatch metrics and alarms
- model package export to tar/S3
- deployment-state driven promote and rollback flows

The current scripts in `pipelines/` now expect model packages and deployment-state inputs instead of SQLite lookups or loose manifest files.

## Validation
Run the full test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
