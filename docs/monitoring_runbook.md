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
