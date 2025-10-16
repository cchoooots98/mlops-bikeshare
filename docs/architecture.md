# System Architecture and Data Flow (Final)

This document captures the implemented architecture for the Bikeshare project: ingestion to inference, monitoring, the Business Dashboard (Step 9), and the Step 10 promotion and rollback flow. It complements `docs/ops_sla.md` and the runbook.

> Region: `ca-central-1` • City: `nyc` • Namespace: `Bikeshare/Model` • Endpoints: `bikeshare-staging`, `bikeshare-prod`

---

## Table of Contents
- [High-level Diagram](#high-level-diagram)
- [Components](#components)
- [Data Flow](#data-flow)
- [Business Dashboard (Step 9)](#business-dashboard-step-9)
- [Metrics and Monitoring](#metrics-and-monitoring)
- [IAM (Least Privilege)](#iam-least-privilege)
- [Step 10: Prod Admission and Cutover](#step-10-prod-admission-and-cutover)
- [Performance and Cost Notes](#performance-and-cost-notes)
- [Appendix: View SQL Skeletons](#appendix-view-sql-skeletons)

---

## High-level Diagram

```
[Sources]
  ├─ GBFS (station info/status)
  ├─ Weather API
  └─ Labels (actual stock/dock events)
        |
        v
[S3 Raw / Bronze]  --->  [ETL / Features (Batch)]  --->  [Training & Registry]
        |                          |                              |
        |                          v                              v
        |                   [Features Offline]            [Model Artifacts]
        |                          |                              |
        v                          |                              |
[Online Inference (SageMaker Endpoint: staging/prod)] <------------+
        |
        +--> Predictions (S3/Athena)
        +--> DataCapture (optional)
        +--> Custom Metrics (CloudWatch, namespace Bikeshare/Model)
        |
        v
[Monitoring and Alarms (CloudWatch)]  --->  [Runbook / On-call]
        |
        v
[Business Dashboard (Streamlit + Athena + CloudWatch)]
```

---

## Components

- **Ingestion and ETL**: fetch GBFS feeds and weather, write to S3 raw; batch ETL builds feature tables.
- **Model Training**: builds model artifacts and registers versions (details in training docs).
- **Online Inference**: SageMaker endpoint (`bikeshare-staging` or `bikeshare-prod`) serves predictions; batch driver emits one heartbeat per 10-min batch.
- **Monitoring**: CloudWatch service metrics plus custom metrics under `Bikeshare/Model` with `{EndpointName, City}`.
- **Dashboard**: Streamlit app reads Athena views and CloudWatch metrics to render business and system health.

---

## Data Flow

1) **Raw ingestion**: `station_information_raw`, `station_status_raw`, `weather_hourly_raw` in S3.  
2) **Feature build**: `features_offline` for training and reference.  
3) **Inference**: the online predictor writes a window of predictions to `inference` (including horizon minutes and probabilities).  
4) **Monitoring**: quality metrics (`PR-AUC-24h`, `F1-24h`), drift (`PSI`), and cadence (`PredictionHeartbeat`) are published to CloudWatch.  
5) **Dashboard**: queries the latest 2 hours of predictions and the last 24 hours of metrics to present the city map, top-N, model/system health, and freshness.

---

## Business Dashboard (Step 9)

### Pages and Interactions
- **City Map**: colored by risk = max(P_bikeout, P_dockout); clicking a station shows details and a 2-hour trajectory.
- **Top-N Risk Stations**: ranking with a simple "suggest move" column (proportional to risk and capacity).
- **Model Health**: time series of `PR-AUC-24h`, `F1-24h`, samples, and PSI.
- **System Health**: `ModelLatency` (avg) and `OverheadLatency` (avg as p95 proxy), 4xx/5xx counts.
- **Data Freshness**: per-table latest timestamps and delay in minutes.

### Data and Metric Sources

**Athena views used by the app**
- `v_station_information`: one row per station with latest info
- `v_predictions`: 0–120 minutes ahead window for the map and charts
- `v_quality` (optional): last 24 hours of quality and labels

**CloudWatch metrics**
- **Custom (`Bikeshare/Model`)** with `{EndpointName, City}`: `PR-AUC-24h`, `F1-24h`, `PSI`, `PredictionHeartbeat`
- **AWS/SageMaker** with `{EndpointName, VariantName=AllTraffic}`: `ModelLatency`, `OverheadLatency`, `Invocation4XXErrors`, `Invocation5XXErrors`

> Note: If you prefer strict p95 in charts, query `ModelLatency` with `Stat = "p95"` via `GetMetricData`.

### App Configuration (secrets)

```toml
region = "ca-central-1"
city = "nyc"
cw_custom_ns = "Bikeshare/Model"
sm_endpoint = "bikeshare-staging"   # switch to "bikeshare-prod" after cutover

aws_profile = "Shirley"
db = "mlops_bikeshare"
workgroup = "primary"
athena_output = "s3://mlops-bikeshare-387706002632-ca-central-1/athena_results/"

view_station_info_latest = "v_station_information"
view_predictions         = "v_predictions"
view_quality             = "v_quality"
```

### Caching and Performance

- Cache AWS clients with `st.cache_resource` and data with `st.cache_data(ttl=60)`.
- Use 5-min periods for `GetMetricData` to align with posting cadence and keep API calls low.
- Keep Athena windows small: predictions 2h, quality 24h; dashboard warm load under 3 seconds.

---

## Metrics and Monitoring

- **Batch-level customs**: publish one `PredictionHeartbeat` per 10-min batch.
- **Quality**: compute and post `PR-AUC-24h` and `F1-24h` on the same 10-min cadence.
- **Drift**: compute and post `PSI` hourly (warn 0.20, critical 0.30).
- **Errors/Latency**: rely on SageMaker metrics for `ModelLatency`, `OverheadLatency`, `Invocation4XXErrors`, `Invocation5XXErrors`.
- **Alarm catalog** lives in `docs/ops_sla.md` (prod names and thresholds).

---

## IAM (Least Privilege)

**Dashboard identity** (local profile or instance profile) needs read-only:
- Athena/Glue read on the `mlops_bikeshare` database and the S3 prefixes it queries
- CloudWatch `GetMetricData`, `ListMetrics`, `GetMetricStatistics`

**CI/CD role** for promotion needs:
- SageMaker: `CreateEndpoint`, `CreateEndpointConfig`, `UpdateEndpoint`, `UpdateEndpointWeightsAndCapacities`, `Describe*`, `List*`
- CloudWatch read for the gate script
- S3 read for model artifacts, configs

---

## Step 10: Prod Admission and Cutover

**Admission gate (tools/check_gate.py)** checks the last 24 hours before promotion:
- `PR-AUC-24h >= 0.70`, `F1-24h >= 0.55`
- `ModelLatency p95 <= 200 ms`, `Invocation5XXErrors = 0`
- `PredictionHeartbeat >= 144`
- `PSI < 0.20` (waiver required if exceeded)

**Promotion (GitHub Actions)**:
- Workflow `promote_prod.yml` runs the gate, creates a fresh EndpointConfig, and updates or creates `bikeshare-prod`, waiting for `InService`.
- For A/B, maintain two variants on the same endpoint and adjust weights with `UpdateEndpointWeightsAndCapacities`.

**Rollback**:
- A/B: set `Baseline=1.0, Candidate=0.0`
- Blue/green: update endpoint back to the previous EndpointConfig
- File an incident and attach CloudWatch graphs; refresh baselines if the candidate is rejected

**Post-cutover tasks**:
- Switch `.streamlit/secrets.toml` to `sm_endpoint = "bikeshare-prod"`
- Disable or downsize staging; verify all prod alarms are green

---

## Performance and Cost Notes

- Prefer 5-min periods and batch-level custom metrics to reduce CloudWatch charges.
- Apply S3 lifecycle to data capture, monitoring outputs, and Athena results.
- Downsize or stop staging when prod is stable; keep prod instance at the smallest size that meets SLO.

---

## Appendix: View SQL Skeletons

```sql
-- Latest station info (one row per station)
CREATE OR REPLACE VIEW mlops_bikeshare.v_station_information AS
WITH info_latest AS (
  SELECT city, station_id, name, capacity, lat, lon,
         row_number() OVER (PARTITION BY city, station_id ORDER BY dt DESC) rn
  FROM mlops_bikeshare.station_information_raw
)
SELECT city, station_id, name, capacity, lat, lon
FROM info_latest WHERE rn = 1;
```

```sql
-- Predictions window (<= 2h ahead for the dashboard)
CREATE OR REPLACE VIEW mlops_bikeshare.v_predictions AS
SELECT station_id, dt, horizon_min,
       CAST(p_bikeout AS double) AS p_bikeout,
       CAST(p_dockout AS double) AS p_dockout
FROM mlops_bikeshare.inference
WHERE city='nyc'
  AND from_iso8601_timestamp(dt) <= current_timestamp + INTERVAL '2' hour;
```

```sql
-- Optional quality join (24h)
CREATE OR REPLACE VIEW mlops_bikeshare.v_quality AS
SELECT *
FROM mlops_bikeshare.monitoring_quality
WHERE city='nyc'
  AND from_iso8601_timestamp(dt) >= current_timestamp - INTERVAL '24' hour;
```
