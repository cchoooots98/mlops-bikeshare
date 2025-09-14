# Monitoring Runbook — Ingestion

## Symptoms
- Athena missing last 2 partitions; CloudWatch `IngestFailures` > 0; Lambda error logs spike.

## Quick Triage
1) Check Lambda logs for the last run (timeout/network/validation error).
2) Check `ingest_errors/` for payload & error summary.
3) Manually re-run Lambda with `{ "city": "nyc" }` test event.
4) Verify S3 write permission and EventBridge rule status.

## Common Causes & Fix
- GBFS endpoint 5xx → rely on EventBridge retry; consider backoff.
- Schema change (new field / type) → update `validators.py`, redeploy.
- S3 permission issue → confirm IAM policy (PutObject, GetObject).

## Bypass Strategy
- Temporarily relax validation to warn-only (Pydantic model to `model_validate(..., strict=False)`).
- Manual backfill: invoke Lambda with historical `last_updated` override (utility script).

## Metrics & Alarms
- `IngestFailures` > 0 (Critical)
- `No new partition in 15 min` (Warning) — via Athena scheduled query or Glue crawler freshness.

---
# Monitoring Runbook — Online Inference & Feedback Loop

## Symptoms
- GitHub Actions workflow `inference_loop.yml` fails.
- No new parquet files appear under `inference/` partitions for more than 10 minutes.
- `monitoring/quality/` partitions are empty or delayed by more than 30 minutes.
- Athena views return errors or missing data.

## Quick Triage
1. Inspect GitHub Actions logs (`Run python -m src.inference.handler`) for IAM or network errors.
2. Check CloudWatch Logs to confirm the SageMaker Endpoint was invoked (`InvokeEndpoint` latency/throughput).
3. In Athena console, run:
   ```sql
   MSCK REPAIR TABLE inference;
   MSCK REPAIR TABLE monitoring_quality;
   ```
   to ensure partitions are up to date.
4. Browse S3 under `s3://<bucket>/inference/city=.../dt=.../` to confirm parquet outputs exist.

## Common Causes & Fix
- **Glue/Athena IAM denial** → Ensure the GitHub OIDC role has `glue:GetDatabase`, `glue:CreateTable`, `glue:GetTable`, `glue:GetPartitions` permissions.
- **Schema mismatch or missing features** → Validate against `src/features/schema.py` FEATURE_COLUMNS.
- **SageMaker Endpoint error** → Review the `raw` column in inference outputs; if needed, test the endpoint directly with boto3.
- **Actuals missing** → The +30 min ground-truth data has not landed yet; the next scheduled run will backfill automatically.

## Bypass Strategy
- Temporarily disable `_quality_table_create_if_absent` to only write inference outputs, avoiding Glue permission blockers.
- Manually backfill actuals:
  ```Powershell
  python src/inference/handler.py --backfill --city nyc --dt 2025-09-12-12-00
  ```
- For older partitions, schedule a nightly backfill job to recompute joins.

## Metrics & Alarms
- **Batch success rate ≥ 99%** (check GitHub Actions run history).
- **Endpoint latency**: CloudWatch `SageMaker/ModelLatency` (target P95 ≤ 200 ms).
- **Data quality**: join rate in `monitoring_quality` ≥ 95%.

## Data Dictionary
- **inference/**
  - `station_id`  
  - `yhat_bikes` (float)  
  - `yhat_bikes_bin` (binary, threshold = 0.15)  
  - `raw` (JSON string from endpoint)  
  - Partition keys: `city`, `dt`

- **monitoring/quality/**
  - `station_id`  
  - `dt` (prediction time)  
  - `dt_plus30` (actual observation time)  
  - `yhat_bikes`, `yhat_bikes_bin`  
  - `y_stockout_bikes_30` (label)  
  - `bikes_t30` (actual bikes)  
  - Partition keys: `city`, `ds`
