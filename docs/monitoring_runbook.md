# Monitoring Runbook — mlops-bikeshare

_Last updated: 2025-10-02 • Region: **ca-central-1** • City: **nyc**_  
_Endpoints: **bikeshare-staging**, **bikeshare-prod**_  
_Routing: SNS topic **sm-alerts** → Email + Slack (verified)_

**Step-8 acceptance status**  
- Drift/Quality/System alarms can trigger and route: **Verified (SNS + forced ALARM drill)**  
- Dashboard trends (24–72h): **Available** on `bikeshare-ops` (quality, hit-rate, latency, errors, heartbeat)  
- MTTA drill: **2.6 minutes** (2025-10-02, Slack/email receipt)

---

## 0) Purpose & Scope
This runbook explains **how to detect, triage, and resolve** issues for:
1) **Ingestion** (GBFS + weather into S3/Glue/Athena)  
2) **Online Inference & Feedback Loop** (predictions + ground-truth)  
3) **Model Monitor (Drift & Data Quality)**  
4) **Endpoint/System Health** (SageMaker latency/errors, Lambda jobs)

It also documents **bypass/rollback** strategies to restore service quickly while the root cause is investigated.

---

## 1) Signals & Where They Live

### Custom metrics — `Bikeshare/Model`
- `PR-AUC-24h`, `F1-24h`, `ThresholdHitRate-24h`, `Samples-24h`
- `PredictionHeartbeat` (batch success/freshness pulse)

### Native metrics — `AWS/SageMaker`
- `ModelLatency` (p50/p95), `OverheadLatency` (p50/p95)
- `Invocations`, `Invocation4XXErrors`, `Invocation5XXErrors`

### Alarms (current names in use)
- **Drift/Quality jobs:** `bikeshare-data-drift-failed`, `bikeshare-data-quality-failed`
- **Batch job health:** `bikeshare-infer-Duration-p95-gt-12m`, `bikeshare-infer-Errors-gt0`, `mlops-bikeshare-lambda-errors`
- **Endpoint/system:** `sm-prod-5XX`, `sm-prod-avg-latency`, `sm-prod-latency`, `one anomaly-band alarm ModelLatency (p95)`
- **Model quality (staging gate):** `staging-f1-low` (recommend also `staging-prauc-low`)

---

## 2) Triage Flow (use this first)

1. **Acknowledge** alert in Slack/email. Open CloudWatch **Alarms** → view state history.  
2. **Classify** the alert:
   - _Data/Ingestion_ • _Drift/Data Quality_ • _Model Quality_ • _Endpoint/System_ • _Pipeline/Lambda_
3. **Open dashboard** `bikeshare-ops` → check last **24–72h** for quality/hit-rate/latency/errors/heartbeat.
4. **Follow the relevant playbook** (sections below).
5. **If impact exists**, apply **bypass/rollback** (see §6) to stabilize; continue root cause.
6. **Close incident** with brief note (symptoms, cause, fix, follow-ups).

---

## 3) Runbook — Ingestion

### Symptoms
- Athena missing last 2 partitions
- CloudWatch metric `IngestFailures` > 0
- Lambda **ingestion** error logs spike

### Quick Triage
1) **Lambda logs:** check last run for timeout/network/validation failures.  
2) **S3 error drops:** browse `ingest_errors/` for payload + error summaries.  
3) **Manual re-run:** invoke Lambda test with `{ "city": "nyc" }`.  
4) **Infra checks:** S3 `PutObject` permissions; EventBridge rule enabled and last trigger time.

### Common Causes & Fix
- **GBFS 5xx / flaky upstream:** rely on EventBridge retry; increase backoff if repeated.  
- **Schema change (new field/type):** update `validators.py`, redeploy function/package.  
- **S3/IAM denial:** verify role has `s3:PutObject`, `s3:GetObject`, `s3:ListBucket`.

### Bypass Strategy
- **Relax validation to warn-only**: call Pydantic with `model_validate(..., strict=False)` temporarily.  
- **Manual backfill:** invoke Lambda with historical `last_updated` override (utility script) to fill gaps.

### Metrics & Alarms
- `IngestFailures` > 0 → **Critical**  
- `No new partition in 15 min` → **Warning** (Athena scheduled query or Glue crawler freshness)

---

## 4) Runbook — Online Inference & Feedback Loop

### Symptoms
- GitHub Actions workflow `inference_loop.yml` fails (or Lambda predictor schedule fails).
- No new parquet under `inference/` for **> 10 minutes**.
- `monitoring/quality/` partitions empty or **> 30 minutes** delayed.
- Athena views error or return empty sets.

### Quick Triage
1) **Workflow/Lambda logs:** look for IAM denials, timeouts, network issues.
