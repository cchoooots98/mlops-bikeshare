# MLOps – Bikeshare (End-to-End)

This repository implements data ingestion, feature engineering, training, real-time inference on Amazon SageMaker, monitoring (CloudWatch + custom metrics), and a Streamlit business dashboard. It reflects the final Step 10 state: a live prod endpoint with promotion gates, rollback, and updated documentation.

> Region: `eu-west-3` • City: `paris` • Custom metrics namespace: `Bikeshare/Model` • Endpoints: `bikeshare-staging`, `bikeshare-prod`

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
region = "eu-west-3"
city = "paris"
cw_custom_ns = "Bikeshare/Model"

# IMPORTANT: Start with staging, switch to prod after Step 10 cutover
sm_endpoint = "bikeshare-staging"

# Athena access
aws_profile = "Shirley"           # optional for local runs
db = "mlops_bikeshare"
workgroup = "primary"
athena_output = "s3://mlops-bikeshare-387706002632-eu-west-3/athena_results/"

# View names used by the app
view_station_info_latest = "v_station_information"
view_predictions         = "v_predictions"
view_quality             = "v_quality"         # optional
```

Metrics must be emitted with dimensions `{ EndpointName, City }` for the custom namespace `Bikeshare/Model`. Batch-level posting is strongly recommended (one datapoint per 10-min batch), not per request.

For ingestion and local Airflow runs, the weather pipeline now uses **OpenWeather One Call 3.0** with a 10-minute cadence. The required runtime settings are:

```powershell
$env:OPENWEATHER_API_KEY = "<your-openweather-api-key>"
$env:RAW_S3_BUCKET = "<your-s3-bucket>"
$env:WEATHER_CITY = "paris"
```

Weather raw payloads land in:

```text
s3://<your-s3-bucket>/raw/weather/city=paris/dt=YYYY-MM-DD-HH-MM/
```

Holiday raw payloads land in:

```text
s3://<your-s3-bucket>/raw/holidays/country=FR/year=YYYY/dt=YYYY-MM-DD-HH-MM/
```

If you use an AWS SSO profile in Docker, mount the SSO cache as writable. A fully read-only `~/.aws` mount can break token refresh during S3 writes.

Weather ingestion now lands in two warehouse staging tables:

- `stg_weather_current`
- `stg_weather_hourly`

After changing `OPENWEATHER_API_KEY` in your shell or `.env`, restart the Airflow containers so the running webserver and scheduler pick up the new value.

dbt then builds the final star-schema weather dimension:

- `dim_weather`

Holiday ingestion writes raw API responses to S3 and loads `stg_holidays`. The date dimension is owned by dbt and built from staging as `analytics.dim_date`.

The warehouse direction is now dbt-first:

- Python ingestion owns raw landing and `public.stg_*` only
- dbt owns curated dimensions such as `dim_weather` and `dim_date`
- dbt `intermediate/` and `features/` layers now own the formal warehouse feature engineering path

Weather feature design now aligns to `dim_weather`. The curated weather contract keeps current-observation fields plus one merged hourly forecast row per ingest bucket:

- current fields: `temperature_c`, `humidity_pct`, `wind_speed_ms`, `precipitation_mm`, `weather_code`, `weather_main`
- merged hourly fields: `hourly_temperature_c`, `hourly_humidity_pct`, `hourly_wind_speed_ms`, `hourly_precipitation_mm`, `hourly_precipitation_probability_pct`, `hourly_weather_code`, `hourly_weather_main`

The merged hourly row is chosen from the latest `forecast_at` within the same `city + run_id + snapshot_bucket_at` bucket, and null hourly fields are backfilled from earlier `forecast_at` rows in that same bucket.

Legacy feature fields such as `temp_c`, `precip_mm`, `wind_kph`, `rhum_pct`, `pres_hpa`, `wind_dir_deg`, `wind_gust_kph`, and `snow_mm` are no longer the target contract.

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
  --city paris `
  --region eu-west-3
```

On CI, the gate runs as the first job in `.github/workflows/promote_prod.yml`.

---

## Promote to Prod (Step 10)

Use the GitHub Actions workflow **"Promote to Prod"**. Inputs:

- `region=eu-west-3`, `city=paris`
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
