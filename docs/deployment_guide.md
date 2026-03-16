# Deployment Guide

## Purpose
This guide is the command authority for:
- local validation
- EC2 always-on deployment
- Terraform infrastructure
- staging deployment
- promotion and rollback

If you are actively building the project, use this together with:
- [current_state_to_enterprise_operator_manual.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/plan_detail/current_state_to_enterprise_operator_manual.md)

## 1. Local Validation
### Pre-checks
- Python is `3.11.x`
- `.env` exists
- Docker Desktop is running

### Commands
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip check
docker compose up -d
.\.venv\Scripts\python.exe -m pytest -q
docker compose ps
```

### Expected output
- `pip check` ends with no dependency conflicts
- `pytest -q` ends with all tests passing
- `docker compose ps` shows `airflow-webserver`, `airflow-scheduler`, `airflow-postgres`, and `dw-postgres` as running

## 2. Local Runtime Contract
### Formal local env
```powershell
$env:PGHOST = "localhost"
$env:PGPORT = "15432"
$env:PGDATABASE = "velib_dw"
$env:PGUSER = "velib"
$env:PGPASSWORD = "velib"
$env:AWS_REGION = "eu-west-3"
$env:CITY = "paris"
$env:BUCKET = "bikeshare-paris-dev"
$env:MLFLOW_TRACKING_URI = "sqlite:///model_dir/mlflow.db"
```

### Validation
```powershell
.\.venv\Scripts\python.exe -c "from src.config import load_runtime_settings; print(load_runtime_settings())"
```

### Expected output
- a printed `RuntimeSettings(...)` object with `pg_host='localhost'`
- `MLFLOW_TRACKING_URI` resolved to `sqlite:///model_dir/mlflow.db`

## 3. Local Orchestration Loop
This phase is only valid if you already have a reachable AWS endpoint.

If you do not have a staging or prod endpoint yet, stop here and finish Section 6 first.

### 3.1 Activate one local deployment state
```powershell
.\.venv\Scripts\python.exe -c "from src.model_package import activate_package; print(activate_package(r'model_dir/packages/bikes/<model-name>/<run-id>', r'model_dir/deployments/bikes/local.json', environment='local', source='manual'))"
```

### 3.2 Point the local loop at a real endpoint
```powershell
$env:TARGET_NAME = "bikes"
$env:SERVING_ENVIRONMENT = "local"
$env:DEPLOYMENT_STATE_PATH = "model_dir/deployments/bikes/local.json"
$env:SM_ENDPOINT = "bikeshare-bikes-staging"
$env:BUCKET = "<your-bucket>"
```

### 3.3 Run predictor
```powershell
.\.venv\Scripts\python.exe -m src.inference.predictor
```

### Expected output
- one or more lines like:
  - `[predictor] wrote <n> rows to s3://<bucket>/inference/target=bikes/city=paris/dt=<dt>/predictions.parquet`

### 3.4 Run quality backfill
```powershell
.\.venv\Scripts\python.exe -m src.monitoring.quality_backfill
```

### Expected output
- a line like:
  - `[quality] wrote <n> shard(s).`

### 3.5 Run metrics dry-run
```powershell
.\.venv\Scripts\python.exe -m src.monitoring.metrics.publish_custom_metrics `
  --bucket <your-bucket> `
  --quality-prefix AUTO `
  --endpoint bikeshare-bikes-staging `
  --city-dimension paris `
  --target-name bikes `
  --environment staging `
  --dry-run
```

### Expected output
- a line beginning with `Computed metrics (last 24h):`
- a line beginning with `Dry run only. Prepared 4 metrics`

## 4. EC2 Always-On Deployment
### Host checklist
- Ubuntu LTS or Amazon Linux
- Docker and Docker Compose installed
- attached instance profile
- persistent disk for Postgres and Airflow logs
- repository plus `.env`

### Required rule
- do not mount a workstation AWS SSO cache into the EC2 stack

### Commands
```bash
git clone <your_repo_url>
cd mlops-bikeshare-202508
cp .env.example .env
docker compose up -d
docker compose ps
curl -I http://localhost:8080
```

### Expected output
- Compose services are healthy
- Airflow responds with `HTTP/1.1 200 OK` or a redirect response

### Required evidence
- 72-hour Airflow success window
- dbt freshness evidence
- successful serving DAG runs for prediction, quality backfill, metrics publish, and PSI publish
- restart recovery evidence

## 5. Terraform Infrastructure
Terraform in this repository manages the long-lived AWS platform only.
Staging deployment, production promotion, and rollback remain separate release steps in Sections 6-8.

### Before you start
- bootstrap the remote backend once from `infra/terraform/bootstrap`
- use the bootstrap outputs when initializing `infra/terraform/live`
- do not treat any repo default `aws_profile` value as canonical; set your own value, use `AWS_PROFILE`, or leave it null
- do not commit secrets in tfvars files
- if a repo default backend or profile does not match your environment, treat it as "must replace", not as an approved default

### Optional variables
- `city`
- `alarm_email_endpoint`

### Commands
```powershell
cd infra\terraform\bootstrap
terraform init
terraform validate
terraform plan
terraform apply
$env:TF_STATE_BUCKET = terraform output -raw tf_state_bucket_name
$env:TF_STATE_REGION = terraform output -raw aws_region
```

Then initialize and apply the long-lived platform stack:

```powershell
cd ..\live
terraform init -reconfigure `
  -backend-config="bucket=$env:TF_STATE_BUCKET" `
  -backend-config="key=infra/live/terraform.tfstate" `
  -backend-config="region=$env:TF_STATE_REGION" `
  -backend-config="encrypt=true"
terraform validate
terraform plan
terraform apply
terraform output
```

### Expected output
- `terraform validate` succeeds
- bootstrap output includes:
  - `tf_state_bucket_name`
  - `aws_region`
- `terraform output` includes:
  - `data_bucket_name`
  - `router_lambda_name`
  - `sns_topic_arn`
  - `cloudwatch_dashboard_name`
  - `sagemaker_role_arn`

### Backend notes
- `infra/terraform/bootstrap` creates only the remote state bucket
- S3 native lockfiles handle state locking for the `live` stack
- no DynamoDB lock table is required for a fresh setup

### Immediate post-apply checks
```powershell
aws sns publish --topic-arn <sns-topic-arn> --message "bikeshare monitoring test" --region eu-west-3
```

Verify in AWS:
- data bucket exists
- ECR repo exists
- router lambda exists
- CloudWatch dashboard exists
- custom metric alarms exist for PR-AUC, F1, PredictionHeartbeat, and PSI
- service alarms exist for latency and 5xx

## 6. AWS Staging Deployment
### 6.1 Build the inference image
```powershell
docker build -f docker/mlflow-pyfunc.Dockerfile -t bikeshare-infer:latest .
docker tag bikeshare-infer:latest <account>.dkr.ecr.eu-west-3.amazonaws.com/mlops-bikeshare:latest
docker push <account>.dkr.ecr.eu-west-3.amazonaws.com/mlops-bikeshare:latest
```

### 6.2 Export packages
```powershell
.\.venv\Scripts\python.exe -m pipelines.export_and_upload_model `
  --package-dir model_dir/packages/bikes/<run-dir> `
  --output-dir dist/model_packages `
  --s3-uri s3://<bucket>/packages/bikes/latest.tar.gz `
  --region eu-west-3

.\.venv\Scripts\python.exe -m pipelines.export_and_upload_model `
  --package-dir model_dir/packages/docks/<run-dir> `
  --output-dir dist/model_packages `
  --s3-uri s3://<bucket>/packages/docks/latest.tar.gz `
  --region eu-west-3
```

### 6.3 Deploy staging endpoints
```powershell
.\.venv\Scripts\python.exe -m pipelines.deploy_staging `
  --endpoint-name bikeshare-bikes-staging `
  --role-arn <role_arn> `
  --image-uri <image_uri> `
  --package-s3-uri s3://<bucket>/packages/bikes/latest.tar.gz `
  --package-dir model_dir/packages/bikes/<run-dir>

.\.venv\Scripts\python.exe -m pipelines.deploy_staging `
  --endpoint-name bikeshare-docks-staging `
  --role-arn <role_arn> `
  --image-uri <image_uri> `
  --package-s3-uri s3://<bucket>/packages/docks/latest.tar.gz `
  --package-dir model_dir/packages/docks/<run-dir>
```

### Expected output
- deployment JSON printed by the CLI
- state files written to:
  - `model_dir/deployments/bikes/staging.json`
  - `model_dir/deployments/docks/staging.json`
- both endpoints become `InService`

## 7. Promotion
### Gate commands
```powershell
.\.venv\Scripts\python.exe -m test.check_gate --endpoint bikeshare-bikes-staging --city paris --region eu-west-3 --target-name bikes --environment staging
.\.venv\Scripts\python.exe -m test.check_gate --endpoint bikeshare-docks-staging --city paris --region eu-west-3 --target-name docks --environment staging
```

### Expected output
- pass:
  - `Admission gate PASSED`
- fail:
  - `Admission gate FAILED:` followed by one or more concrete failed metrics

### Promote only after the gate passes
```powershell
.\.venv\Scripts\python.exe -m pipelines.promote `
  --source-deployment-state-path model_dir/deployments/bikes/staging.json `
  --target-deployment-state-path model_dir/deployments/bikes/production.json `
  --target-environment production

.\.venv\Scripts\python.exe -m pipelines.promote `
  --source-deployment-state-path model_dir/deployments/docks/staging.json `
  --target-deployment-state-path model_dir/deployments/docks/production.json `
  --target-environment production
```

### Verify after promotion
- `production.json` points to the expected package, run ID, and endpoint
- dashboard reflects the expected prod version

## 8. Rollback
### Pre-check
- `previous_prod.json` exists for the impacted target

### Command
```powershell
.\.venv\Scripts\python.exe -m pipelines.rollback `
  --target-name bikes `
  --environment production `
  --from-state model_dir/deployments/bikes/production.json `
  --to-state model_dir/deployments/bikes/previous_prod.json
```

### Expected output
- JSON containing:
  - `target_name`
  - `environment`
  - `active_state_path`
  - `rollback_source_path`
  - `restored_deployment_state_path`
  - `previous_run_id`
  - `restored_run_id`

### Verify after rollback
- only one target changed
- endpoint name and package reference match the previous approved state
- dashboard and CloudWatch align with the restored state
