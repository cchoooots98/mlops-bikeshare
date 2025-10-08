# System Architecture and Data Flow 

This document updates **docs/architecture.md** to capture the **implemented Business Dashboard (Step 9)**: its pages, data sources, metrics mapping, IAM, and performance/cost considerations. It complements the existing architecture sections (ingestion, training, deployment, monitoring) and is meant to be read alongside `docs/ops_sla.md` and `docs/monitoring_runbook.md`.

---

## 2.2 Business Dashboard (Step 9) — Implemented

### Pages & Interactions
- **1) City Map** (Folium / OpenStreetMap)
  - Color by **risk = max(P_bikeout, P_dockout)**.
  - Click a station ⇒ sidebar shows **station info** and **2‑hour forecast trajectory** (minutes ahead vs probability).
  - Optional **radius filter** (Haversine) to restrict the map and ranking to a neighborhood.
- **2) Top‑N Risk Stations**
  - Sort by `risk` and filter by radius; display a simple **rebalancing suggestion**: 
    [`suggest_move` = round(β × capacity × risk)], default β = 0.3.
- **3) Model Health**
  - Time‑series: **PR‑AUC‑24h**, **F1‑24h**, **ThresholdHitRate‑24h**, **Samples‑24h**, **PSI** (CloudWatch custom metrics).
  - Status badges: endpoint InService, existence of required custom metrics.
- **4) System Health**
  - Time‑series: **ModelLatency** (avg ≈ p50 proxy), **OverheadLatency** (avg used as **p95 proxy**), **Invocation4XXErrors**, **Invocation5XXErrors** (CloudWatch AWS/SageMaker namespace).
  -  Batch success: **PredictionHeartbeat** custom metric (1 per successful batch) or **LambdaSuccessRate**.
- **5) Data Freshness**
  - Table of latest partition timestamps and **delay (min)** across: `station_status_raw`, `station_information_raw`, `weather_hourly_raw`, `features_offline`, `inference`, `monitoring_quality`.

> **Acceptance (implemented):** Page load under ~3s on warm cache; all tiles render without blanks; filters and map clicks remain responsive.

---

### Data & Metric Sources (authoritative mapping)

#### Athena (SQL / Views)
- **`v_station_information`** → one row per station: `station_id, name, capacity, lat, lon` (latest).
- **`v_predictions`** → 0–120‑minute horizon window for the dashboard, expected columns: 
  `station_id, dt (ISO), horizon_min, p_bikeout, p_dockout`.
- **`v_quality`** (optional) → 24h join for quality/labels to enrich model‑health views.

> **Tables referenced for freshness:** `station_status_raw`, `station_information_raw`, `weather_hourly_raw`, `features_offline`, `inference`, `monitoring_quality`.

#### CloudWatch (Metrics)
- **Custom (namespace `Bikeshare/Model`)** — dimensions: `{ EndpointName, City }`
  - `PR-AUC-24h` (Average), `F1-24h` (Average), `ThresholdHitRate-24h` (Average), `Samples-24h` (Sum or Average), `PredictionHeartbeat` (Sum).
- **AWS/SageMaker** — dimensions: `{ EndpointName, VariantName=AllTraffic }`
  - `ModelLatency` (Average ~ p50 proxy), `OverheadLatency` (Average ~ p95 proxy), `Invocation4XXErrors` (Sum), `Invocation5XXErrors` (Sum).

> **Note on p95:** For strict p95 latency, the app can query `ModelLatency` with `Stat = p95` in `GetMetricData`. The current implementation uses `ModelLatency (avg)` and `OverheadLatency (avg)` as light‑weight proxies to reduce call volume and keep the page under 3s.

---

### Streamlit App Architecture (high‑level)

- **Config** via `.streamlit/secrets.toml`:
  ```toml
  aws_profile = "Shirley"
  region = "ca-central-1"
  db = "mlops_bikeshare"
  workgroup = "primary"
  athena_output = "s3://mlops-bikeshare-387706002632-ca-central-1/athena_results/"
  city = "nyc"

  cw_custom_ns = "Bikeshare/Model"
  sm_endpoint = "bikeshare-staging"

  view_station_info_latest = "v_station_information"
  view_predictions         = "v_predictions"
  view_quality             = "v_quality"        # optional

  tbl_station_info_raw     = "station_information_raw"
  tbl_station_status_raw   = "station_status_raw"
  tbl_weather_hourly_raw   = "weather_hourly_raw"
  tbl_features             = "features_offline"
  tbl_inference            = "inference"
  tbl_monitoring_quality   = "monitoring_quality"
  ```
- **Caching for performance**
  - `@st.cache_resource` for **Athena** and **CloudWatch** clients.
  - `@st.cache_data(ttl=60)` for common SQL queries and metric pulls.
  - CloudWatch `GetMetricData` period = **300s** to match 5‑min cadence and minimize API calls.
- **Latency targets**
  - First warm load ≲ 3s; chart interactions ≲ 500ms; Athena views constrained to 2h (pred) and 24h (quality).

---

### IAM for Dashboard (least privilege)

Grant the Streamlit runtime identity **read‑only** access:
- **Athena/Glue (DB‑scoped)**: `athena:GetQueryResults`, `athena:StartQueryExecution`, `glue:Get*` for the `mlops_bikeshare` DB; S3 read to `athena_results/` and data prefixes (`raw/`, `features/`, `inference/`, `monitoring/`).
- **CloudWatch**: `cloudwatch:GetMetricData`, `cloudwatch:ListMetrics`, `cloudwatch:GetMetricStatistics`.

(Optional) If deploying Streamlit on EC2 or ECS, attach the role via instance/task profile; for local usage, rely on your AWS profile/SSO.

---

### Known Deviations & TODOs (tracked)
- **PSI/KS visualization**: Alarms exist for feature drift (PSI) but the dashboard currently shows only model‑level KPIs. Add a small chart from `Bikeshare/Model → PSI` (24h) and/or surface the latest Model Monitor drift report (statistic/constraints JSON) with a green/amber/red badge.
- **Batch success chart**: If not shown, plot `PredictionHeartbeat` (Sum or Average) to visualize 10‑min predict cadence; retain the `stg-batch-success-rate-crit` alarm as the gate.
- **Feature freshness**: `features_offline` can show stale (by design) unless rebuilt. Keep it informational; it does not block online inference. Consider weekly rebuild or hide from freshness if not maintained.
- **Top‑N table units**: Display `risk` consistently as fraction (0–1) or percent (0–100%); the app currently labels `risk_pct` — ensure formatting and column name match.
- **Strict p95**: Optionally switch System Health to true p95 via `Stat = "p95"` for `ModelLatency`.

---

### How this section aligns to SLOs & Runbook
- Model KPIs (PR‑AUC/F1/ThresholdHitRate/Samples) ↔ **quality SLOs**; 
- Endpoint latency/error metrics ↔ **system SLOs** (p50/p95, 4xx/5xx); 
- Freshness table ↔ **data SLO** and backfill checks; 
- All graphs use 24h windows for at‑a‑glance triage; alarms route per `monitoring_runbook.md`.

---

## Appendix: Quick SQL skeletons (for reproducibility)

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
-- Predictions window (≤ 2h ahead for the dashboard)
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

---

## Revision Log (Step 9)
- Added **2.2 Business Dashboard (Step 9) — Implemented** with: page definitions, data/metric mappings, Streamlit app architecture, IAM, performance notes, and known deviations.
- Included SQL skeletons to reproduce the views used by the dashboard.
