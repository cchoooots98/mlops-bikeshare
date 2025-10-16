# MLOps – Bikeshare (End-to-End)

This repository implements data ingestion, feature engineering, training, real-time inference on Amazon SageMaker, monitoring (CloudWatch + custom metrics), and a Streamlit business dashboard. It reflects the final Step 10 state: a live prod endpoint with promotion gates, rollback, and updated documentation.

> Region: `ca-central-1` • City: `nyc` • Custom metrics namespace: `Bikeshare/Model` • Endpoints: `bikeshare-staging`, `bikeshare-prod`

---

## Table of Contents
- [MLOps – Bikeshare (End-to-End)](#mlops--bikeshare-end-to-end)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Quickstart](#quickstart)
  - [Configuration](#configuration)
  - [Run the Dashboard Locally](#run-the-dashboard-locally)
  - [Monitoring and Alarms](#monitoring-and-alarms)
  - [Admission Gate (Step 10)](#admission-gate-step-10)
  - [Promote to Prod (Step 10)](#promote-to-prod-step-10)
  - [Rollback](#rollback)
  - [Cost Control](#cost-control)
  - [Public vs Private App](#public-vs-private-app)
  - [Repository Layout (high level)](#repository-layout-high-level)
  - [License](#license)

---

## Prerequisites

- Windows 10/11, VS Code, PowerShell
- Python 3.11
- AWS CLI v2 with credentials (SSO or static keys)
- Optional: Docker Desktop if you build containers locally

---

## Quickstart

```powershell
# 1) Create and activate a virtual environment
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt
```

---

## Configuration

The dashboard and utilities read settings from `.streamlit/secrets.toml`:

```toml
# .streamlit/secrets.toml
region = "ca-central-1"
city = "nyc"
cw_custom_ns = "Bikeshare/Model"

# IMPORTANT: Start with staging, switch to prod after Step 10 cutover
sm_endpoint = "bikeshare-staging"

# Athena access
aws_profile = "Shirley"           # optional for local runs
db = "mlops_bikeshare"
workgroup = "primary"
athena_output = "s3://mlops-bikeshare-387706002632-ca-central-1/athena_results/"

# View names used by the app
view_station_info_latest = "v_station_information"
view_predictions         = "v_predictions"
view_quality             = "v_quality"         # optional
```

Metrics must be emitted with dimensions `{ EndpointName, City }` for the custom namespace `Bikeshare/Model`. Batch-level posting is strongly recommended (one datapoint per 10-min batch), not per request.

---

## Run the Dashboard Locally

```powershell
# From repo root
streamlit run app/dashboard.py
```

Once you have promoted to prod, change the endpoint in `.streamlit/secrets.toml`:
```toml
sm_endpoint = "bikeshare-prod"
```

---

## Monitoring and Alarms

- **Custom metrics (namespace `Bikeshare/Model`)** with `{EndpointName, City}`: `PR-AUC-24h`, `F1-24h`, `PSI`, `PredictionHeartbeat`.
- **SageMaker metrics (`AWS/SageMaker`)** with `{EndpointName, VariantName=AllTraffic}`: `ModelLatency`, `OverheadLatency`, `Invocation4XXErrors`, `Invocation5XXErrors`.
- Prod alarm names and thresholds are documented in `docs/ops_sla.md` (final).

---

## Admission Gate (Step 10)

The promotion gate is enforced by `test/check_gate.py`. It queries the last 24 hours and fails the pipeline if any condition is violated:

- `PR-AUC-24h >= 0.70`
- `F1-24h >= 0.55`
- `ModelLatency p95 <= 200 ms`
- `Invocation5XXErrors = 0`
- `PredictionHeartbeat >= 144` (10-min cadence x 24h)
- `PSI < 0.20` (warning threshold; require waiver if exceeded)

Example local run:

```powershell
python test/check_gate.py `
  --endpoint bikeshare-staging `
  --city nyc `
  --region ca-central-1
```

On CI, the gate runs as the first job in `.github/workflows/promote_prod.yml`.

---

## Promote to Prod (Step 10)

Use the GitHub Actions workflow **"Promote to Prod"**. Inputs:

- `region=ca-central-1`, `city=nyc`
- `staging_endpoint=bikeshare-staging`
- `prod_endpoint=bikeshare-prod`
- `model_name=<your-trained-model>`
- `instance_type=ml.m5.large`
- Optional: A/B canary by creating two variants and adjusting weights

The workflow creates a fresh EndpointConfig and updates or creates the `bikeshare-prod` endpoint, then waits for `InService`.

---

## Rollback

If 5xx>0, p95>300 ms, or PR-AUC/F1 drop below thresholds:

1) **A/B rollback**: `UpdateEndpointWeightsAndCapacities` to restore `Baseline=1.0, Candidate=0.0`  
2) **Blue/green rollback**: `UpdateEndpoint` back to the previous EndpointConfig  
3) Record the incident, attach graphs, and refresh baselines if the candidate is rejected

---

## Cost Control

- Stop or downsize **staging** once prod is stable
- Keep metrics at **5-min periods** and **batch-level posting**
- S3 lifecycle for `datacapture/`, `monitoring/`, and `athena_results/`
- Athena: partition and compress to reduce scanned bytes

---

## Public vs Private App

The app runs locally by default. To publish, host it on a small EC2/ALB or on Streamlit Community Cloud and plan for ongoing cost. For portfolio purposes, screenshots and a short demo video are sufficient if you prefer to avoid hosting expense.

---

## Repository Layout (high level)

```
app/                     # Streamlit dashboard
docs/                    # ops_sla.md, architecture.md, runbook, cost 
src/                     # pipelines, monitoring, metrics publishers, utilities
test/                   # check_gate.py and helpers
.github/workflows/       # CI/CD (promote_prod.yml)
```

---

## License

MIT (or your preferred license).
