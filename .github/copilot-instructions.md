# AI agent quickstart for this repo

Purpose: minimal but production-grade MLOps loop to predict 30‑minute bikeshare stockouts using AWS (SageMaker, S3, Athena/Glue), with CI/CD, online inference, and monitoring.

## Big picture
- Data → Features → Train → Register/Package → Deploy Endpoint (staging→prod) → Online Inference → Capture → Monitoring.
- Region/account: ca-central-1, AWS Account 387706002632. Endpoints: repo vars `ENDPOINT_STAGING`, `ENDPOINT_PROD`.
- Infra as code: Terraform in `infra/terraform` (remote S3/DynamoDB backend; see `README.md` quickstart).

## Key directories/files
- Features: `src/features/` — schema in `schema.py` (authoritative `FEATURE_COLUMNS`, `LABEL_COLUMNS`, `REQUIRED_BASE`, dt format `YYYY-MM-DD-HH-mm` UTC). Keep offline/online parity.
- Training: `src/training/train.py` — time-based split, anti-leakage `--gap-minutes`, primary metric PR‑AUC, MLflow artifacts; default experiment `bikeshare-step4`.
- Inference (online): `src/inference/` — `featurize_online.py` builds latest features; `predictor.py` invokes SageMaker endpoint, writes `s3://…/inference/city=…/dt=…/predictions.parquet` with columns `[station_id, dt, yhat_bikes, yhat_bikes_bin, inferenceId, raw]` (threshold 0.15). Lambda entrypoint: `lambda_handler.py`.
- Deploy: `pipelines/deploy_via_sagemaker_sdk.py` — creates Model + EndpointConfig (DataCapture enabled) + upserts Endpoint. Validates ECR image region. NOTE: DataCapture S3 URI is hard‑coded to bucket `mlops-bikeshare-387706002632-ca-central-1` — update if you fork.
- Docker image for MLflow pyfunc: `docker/mlflow-pyfunc.Dockerfile` — copies `src/features/schema.py` into image to guarantee schema parity; exposes `:8080` and supports `/invocations`.
- CI/CD: `.github/workflows/{ci.yml, cd_staging.yml, promote_prod.yml}` — OIDC to AWS, deploys via the deploy script, smoke‑tests with `test/smoke_invoke.py`.

## Conventions and patterns (do this here)
- Always import feature names from `src/features/schema.py`. Preserve exact order when building payloads; models expect MLflow pandas `dataframe_split` with `columns: FEATURE_COLUMNS`.
- Timestamps are 5‑min buckets as strings `YYYY-MM-DD-HH-mm` in UTC; training uses temporal split over distinct `dt` values plus a gap to avoid leakage.
- MLflow tracking defaults to a local SQLite file (see `src/training/train.py`). Workflows set `MLFLOW_TRACKING_URI` explicitly; respect existing env if present.
- Online inference uses per‑row `InferenceId = f"{dt}_{station_id}"` so SageMaker DataCapture can join requests/responses later.
- S3 layout examples: `inference/city=<city>/dt=<dt>/predictions.parquet`, monitoring outputs under `monitoring/` (see docs).

## Essential commands (Windows PowerShell)
- Activate venv: `. .\.venv\Scripts\Activate.ps1`
- Run tests/lint fast: `pytest -q`; `ruff check src pipelines test --fix`; `black --check src pipelines test`
- Train example (adjust window/DB):
  `python -m src.training.train --city nyc --start "2025-08-18 00:00" --end "2025-08-25 23:55" --database mlops_bikeshare --athena-output "s3://mlops-bikeshare-387706002632-ca-central-1/athena/results/" --region ca-central-1 --label y_stockout_bikes_30 --model-type xgboost --valid-ratio 0.2 --gap-minutes 60 --beta 2.0 --experiment bikeshare-step4`
- Deploy (staging) from CI/CD or locally mirror the workflow call:
  `python pipelines/deploy_via_sagemaker_sdk.py --endpoint-name <name> --role-arn <SM_EXECUTION_ROLE_ARN> --image-uri <ECR_URI> --model-data <s3://…/model.tar.gz> --instance-type ml.m5.large --region ca-central-1`
- Smoke invoke endpoint: `python test/smoke_invoke.py --endpoint-name <name> --region ca-central-1`

## CI/CD essentials
- Variables needed (GitHub): `AWS_REGION, AWS_ACCOUNT_ID, ECR_IMAGE_URI, S3_MODEL_TAR, ENDPOINT_STAGING, ENDPOINT_PROD, INSTANCE_STAGING, INSTANCE_PROD`.
- Secrets: `AWS_ROLE_TO_ASSUME`, `SM_EXECUTION_ROLE_Arn`.
- Workflows perform: lint/test/security → deploy to staging on main → manual promote to prod with rollback to previous EndpointConfig on failure.

## Gotchas
- ECR image region must match endpoint region (enforced in deploy script). Build for linux/amd64 for m5.*.
- Keep container deps compatible: image pins `numpy==1.26.*`, `scikit-learn==1.7.*`, `mlflow==3.3.2`.
- If you change feature schema, update both training and online code and re‑build container so `/opt/ml/code/src/features/schema.py` stays in sync.
- Destination S3 for DataCapture must exist and allow PutObject.

## Pointers to deeper docs
See `docs/architecture.md`, `docs/training_eval.md`, `docs/feature_store.md`, and `docs/monitoring_runbook.md` for rationale, metrics, and operations.
