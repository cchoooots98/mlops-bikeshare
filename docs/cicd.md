# CI And CD

## CI
CI should validate:
- unit and integration tests
- feature-contract regressions
- package/deployment-state workflows
- AWS deploy wrappers with fake clients

Recommended local command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Packaging Flow
The deployable unit is now a model package, not a loose MLflow stage reference.

Expected flow:
1. train candidate
2. register candidate
3. assemble/export package tar
4. upload tar to S3
5. deploy package to staging
6. update staging deployment-state record
7. promote deployment state after gate checks

Production scheduling does not run in GitHub Actions.
Formal recurring execution lives on EC2-hosted Airflow.
Staging gate automation and production automation are separate Airflow DAG sets:
- `staging_*` DAGs drive the 24-hour staging gate before promotion
- `serving_*` DAGs drive post-promotion production prediction, quality backfill, metrics, and PSI

## Main Scripts
- `pipelines/export_and_upload_model.py`
  - inputs: package dir or MLflow run/version
  - outputs: local tar and optional S3 URI
- `pipelines/deploy_via_sagemaker_sdk.py`
  - inputs: package tar S3 URI plus optional local package dir
  - outputs: SageMaker deploy result and optional deployment-state JSON
- `pipelines/promote.py`
  - inputs: source deployment state and target deployment state
  - outputs: promoted deployment-state JSON
- `pipelines/rollback.py`
  - inputs: active deployment state and previous approved state
  - outputs: restored deployment-state JSON

## Runtime Inputs
Core runtime inputs are now:
- `MODEL_PACKAGE_DIR`
- `DEPLOYMENT_STATE_PATH`
- `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`
- `CITY`, `BUCKET`, `TARGET_NAME`, `SERVING_ENVIRONMENT`
