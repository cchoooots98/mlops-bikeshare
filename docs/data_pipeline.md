# Data Pipeline

## Overview

The data pipeline covers the full path from raw ingestion to model-ready features:

1. **Ingestion** — Python tasks poll external APIs (GBFS, OpenWeather, holiday APIs) on scheduled cadences.
2. **Raw S3** — Valid raw payloads land in partitioned S3 paths under `raw/`.
3. **Staging (`stg_*`)** — Python normalizes each raw payload into typed staging tables in the `public` schema.
4. **dbt warehouse** — dbt curates staging into dimensions and fact tables in the `analytics` schema.
5. **Feature tables** — dbt intermediate and feature models produce `feat_station_snapshot_5min` and `feat_station_snapshot_latest`, which Python training and online prediction consume directly.

Cross-section relationships:
- The data contract (Section 1) defines what Python is allowed to write and what it must not touch.
- The warehouse model (Section 2) defines the full table inventory and column-level contracts for both staging and curated layers.
- The feature store (Section 3) defines how the curated warehouse tables are assembled into ML-ready features and the alignment rules that keep offline and online paths consistent.

---

## 1. Data Contract

### Scope

- GBFS ingestion runs on two cadences:
  - `station_status`: every 5 minutes
  - `station_information`: daily
- Weather ingestion runs every 10 minutes.
- Holiday ingestion runs yearly or on manual replay.
- All raw timestamps are stored in UTC.
- Raw payloads are partitioned by city and `dt=YYYY-MM-DD-HH-MM`.

### Landing Paths

- `s3://<bucket>/raw/station_information/city=<city>/dt=YYYY-MM-DD-HH-MM/`
- `s3://<bucket>/raw/station_status/city=<city>/dt=YYYY-MM-DD-HH-MM/`
- `s3://<bucket>/raw/weather/city=<city>/dt=YYYY-MM-DD-HH-MM/`
- `s3://<bucket>/raw/holidays/country=<country_code>/year=<year>/dt=YYYY-MM-DD-HH-MM/`

Raw weather JSON remains in S3 for replay and auditing beyond the immediate ingestion window.

### Required Raw Payloads

**station_information**:
- `data.stations[].station_id`
- `data.stations[].name`
- `data.stations[].capacity`
- `data.stations[].lat`
- `data.stations[].lon`

**station_status**:
- `data.stations[].station_id`
- `data.stations[].num_bikes_available`
- `data.stations[].num_docks_available`
- `data.stations[].last_reported`

**OpenWeather current block**:
- `current.dt`
- `current.temp`
- `current.humidity`
- `current.wind_speed`
- `current.weather`

**OpenWeather hourly block**:
- `hourly[].dt`
- `hourly[].temp`
- `hourly[].humidity`
- `hourly[].wind_speed`
- `hourly[].pop`
- optional `hourly[].rain.1h` / `hourly[].snow.1h`
- `hourly[].weather`

**Holiday API block**:
- JSON object of `YYYY-MM-DD -> holiday_name`

### Ingestion Boundary

Python ingestion handles:
- API calls
- raw payload validation
- raw S3 writes
- normalized staging writes into `public.stg_*`

Python ingestion does not own:
- `analytics.dim_weather`
- `analytics.dim_date`
- future curated, intermediate, or feature tables

Python weather ingestion stores only current observations and the next 60 minutes of hourly forecast rows. It does not collapse hourly forecast rows into a single contract row — that merge is dbt's responsibility (see Section 2, Weather Model Contract).

### Transformation Boundary

- Python ingestion handles raw ingestion and normalized staging inserts only.
- dbt handles station SCD2 logic and builds `dim_station`.
- dbt handles base station fact logic and builds `fct_station_status`.
- dbt handles weather summarization and builds `dim_weather`.
- dbt handles holiday/date logic and builds `dim_date`.
- Later machine-learning feature builds may continue to evolve, but the warehouse contract should stay explicit and documented.

### Validation Rules

- Raw payload validation happens in ingestion code before the S3 write and before the staging insert.
- Invalid payloads fail the task and prevent partial structured writes.
- Station duplicate protection is keyed by `city + snapshot_bucket_at` for both station staging tables.
- Weather duplicate protection is keyed by `city + snapshot_bucket_at` in `stg_weather_current`.
- Holiday duplicate protection is keyed by `country_code + holiday_date` within the yearly load.
- Cross-table station inventory sanity is enforced after capacity is available from `dim_station`, not in raw-facing staging.
- `fct_station_status` rejects station snapshots where `num_bikes_available + num_docks_available > station_inventory_capacity_multiplier * capacity`; the default multiplier is `2` and downstream models inherit that filter.

### Error Handling

- Reject bad batches and log task failure in Airflow.
- Persist only valid raw payloads to `raw/...`.
- Investigate schema drift if OpenWeather stops returning `current.dt` or `hourly[]`.
- Holiday ingestion no longer creates or updates `dim_date`; dbt is responsible for rebuilding it from staging.

### Latency SLO

- GBFS raw landing: <= 3 minutes end-to-S3.
- Weather raw landing: <= 10 minutes from scheduled bucket.

---

## 2. Warehouse Data Model

### Scope

This section defines the dbt-first warehouse boundary for GBFS, weather, and holiday data.

### Layer Inventory

#### Raw (S3)

- `raw/station_information/...`
- `raw/station_status/...`
- `raw/weather/...`
- `raw/holidays/...`

#### Python Ingestion to `public` schema

- `stg_station_information`
- `stg_station_status`
- `stg_weather_current`
- `stg_weather_hourly`
- `stg_holidays`

#### dbt Curated in `analytics` schema

- `dim_station`
- `dim_time`
- `dim_weather`
- `dim_date`
- `fct_station_status`

`dim_date` is owned by dbt. Holiday ingestion stops at `stg_holidays`.

#### dbt Intermediate Layer

- `int_station_neighbors`
- `int_station_status_enriched`
- `int_station_weather_aligned`
- `int_station_rollups`

#### dbt Feature Layer

- `feat_station_snapshot_5min`
- `feat_station_snapshot_latest`

These are the formal dbt producer tables that Python training and online prediction consume. See Section 3 for feature alignment rules.

### Station Model Contract

Station data retains `city` from raw landing through staging, dimensions, and facts. `station_id` alone is not a safe long-term business key once the warehouse holds more than one city.

#### `stg_station_information`

- grain: one row per `city + snapshot_bucket_at + station_id`
- audit-only staging columns: `run_id`, `ingested_at`, `source_last_updated`
- columns:
  - `run_id`
  - `ingested_at`
  - `source_last_updated`
  - `city`
  - `snapshot_bucket_at`
  - `station_id`
  - `station_name`
  - `latitude`
  - `longitude`
  - `capacity`

#### `stg_station_status`

- grain: one row per `city + snapshot_bucket_at + station_id`
- audit-only staging columns: `run_id`, `ingested_at`, `source_last_updated`
- columns:
  - `run_id`
  - `ingested_at`
  - `source_last_updated`
  - `city`
  - `snapshot_bucket_at`
  - `station_id`
  - `last_reported_at`
  - `num_bikes_available`
  - `num_docks_available`
  - `is_renting`
  - `is_returning`

#### `dim_station`

- SCD2 dimension with one row per station version
- durable business key: `station_key` = `city + station_id`
- row key: `station_version_key`
- version boundaries come from station information `snapshot_bucket_at`
- tracked attributes: `station_name`, `latitude`, `longitude`, `capacity`
- columns:
  - `station_version_key`
  - `station_key`
  - `city`
  - `station_id`
  - `station_name`
  - `latitude`
  - `longitude`
  - `capacity`
  - `valid_from_utc`
  - `valid_to_utc`
  - `is_current`

#### `fct_station_status`

- lean fact table with one row per `city + snapshot_bucket_at + station_id`
- joins station via `station_version_key`
- columns:
  - `fact_station_status_key`
  - `city`
  - `station_id`
  - `station_key`
  - `station_version_key`
  - `snapshot_bucket_at_utc`
  - `snapshot_bucket_at_paris`
  - `date_id`
  - `time_id`
  - `last_reported_at_utc`
  - `last_reported_at_paris`
  - `num_bikes_available`
  - `num_docks_available`
  - `is_renting`
  - `is_returning`
- intentionally excludes `capacity`, utilization fields, and weather columns

### Weather Model Contract

`dim_weather` is the single source of truth for weather features. It is built from `stg_weather_current` and `stg_weather_hourly`. For each current-weather row, dbt joins hourly rows from the same `city + run_id + snapshot_bucket_at`, takes the latest `forecast_at` as the reference hourly row, and backfills null hourly fields from earlier `forecast_at` rows in that same ingest bucket.

#### `stg_weather_current`

- grain: one row per `city + snapshot_bucket_at`
- columns:
  - `run_id`
  - `ingested_at`
  - `source_last_updated`
  - `city`
  - `snapshot_bucket_at`
  - `observed_at`
  - `temperature_c`
  - `humidity_pct`
  - `wind_speed_ms`
  - `precipitation_mm`
  - `weather_code`
  - `weather_main`
  - `weather_description`
  - `source`

#### `stg_weather_hourly`

- grain: one row per `city + snapshot_bucket_at + forecast_at`
- only includes forecast rows within the next 60 minutes of the matching `stg_weather_current.observed_at`
- columns:
  - `run_id`
  - `ingested_at`
  - `source_last_updated`
  - `city`
  - `snapshot_bucket_at`
  - `observed_at`
  - `forecast_at`
  - `forecast_horizon_min`
  - `temperature_c`
  - `humidity_pct`
  - `wind_speed_ms`
  - `precipitation_mm`
  - `precipitation_probability_pct`
  - `weather_code`
  - `weather_main`
  - `weather_description`
  - `source`

#### `dim_weather`

- columns:
  - `weather_key`
  - `city`
  - `observed_at`
  - `temperature_c`
  - `humidity_pct`
  - `wind_speed_ms`
  - `precipitation_mm`
  - `weather_code`
  - `weather_main`
  - `weather_description`
  - `hourly_forecast_at`
  - `hourly_temperature_c`
  - `hourly_humidity_pct`
  - `hourly_wind_speed_ms`
  - `hourly_precipitation_mm`
  - `hourly_precipitation_probability_pct`
  - `hourly_weather_code`
  - `hourly_weather_main`
  - `source`
  - `snapshot_bucket_at_utc`

### Holiday and Date Logic

- Holiday raw API responses are stored in S3 under `raw/holidays/country=.../year=.../dt=.../`.
- `stg_holidays` stores one row per holiday date for a country/year load; grain is `country_code + holiday_date`.
- columns: `country_code`, `holiday_date`, `is_holiday`, `holiday_name`
- dbt builds `dim_date` from `stg_holidays` and available station-status date bounds.
- dbt owns `is_weekend`, `is_holiday`, and `holiday_name` on `dim_date`.
- Weekend and holiday attributes belong to `analytics.dim_date`, not to Python ingestion.

### Weather Feature Direction

The implemented dbt feature layer (see Section 3) consumes the following weather columns from `dim_weather`:

Active feature columns:
- `temperature_c`
- `humidity_pct`
- `wind_speed_ms`
- `precipitation_mm`
- `weather_code`
- `hourly_temperature_c`
- `hourly_humidity_pct`
- `hourly_wind_speed_ms`
- `hourly_precipitation_mm`
- `hourly_precipitation_probability_pct`
- `hourly_weather_code`

Deprecated / legacy fields (remove in the next feature migration):
- `temp_c`
- `precip_mm`
- `wind_kph`
- `rhum_pct`
- `pres_hpa`
- `wind_dir_deg`
- `wind_gust_kph`
- `snow_mm`

### Diagram

Mermaid source: [diagrams/day3_star_schema.mmd](diagrams/day3_star_schema.mmd)

---

## 3. Feature Store

### Alignment Between Offline and Online

- Time granularity: all bike-station features are aligned on 5-minute `dt`.
- Window definitions: the same rolling windows (15/30/60 minutes) are intended for offline and online use.
- Stockout threshold: 2 bikes/docks.
- Weather source: weather originates from OpenWeather One Call 3.0 raw payloads, lands in warehouse staging tables (see Section 1, Landing Paths), and is summarized into `dim_weather` by dbt (see Section 2, Weather Model Contract).
- Neighbor strategy: K=5 nearest neighbors within radius 0.8 km, weighted by `1/distance`.
- Ownership: dbt is the formal producer of warehouse feature tables; Python is the consumer of final feature tables.

### Weather Features Exposed to the Feature Layer

The curated `dim_weather` dimension exposes one row per weather observation time. The feature layer selects the following columns as first-class model inputs:

- `temperature_c`
- `humidity_pct`
- `wind_speed_ms`
- `precipitation_mm`
- `weather_code`
- `hourly_temperature_c`
- `hourly_humidity_pct`
- `hourly_wind_speed_ms`
- `hourly_precipitation_mm`
- `hourly_precipitation_probability_pct`
- `hourly_weather_code`

Additional columns available in `dim_weather` but not promoted to model features:
- `weather_description`
- `hourly_forecast_at`
- `hourly_weather_main`

Legacy weather feature fields to remove in the next feature migration:
- `temp_c`
- `precip_mm`
- `wind_kph`
- `rhum_pct`
- `pres_hpa`
- `wind_dir_deg`
- `wind_gust_kph`
- `snow_mm`

### Implemented Feature Flow

Current weather processing flow (ingestion through feature layer):

```
raw weather JSON (S3)
  → stg_weather_current  (Python, public schema)
  → stg_weather_hourly   (Python, public schema)
  → dim_weather          (dbt, analytics schema)
  → int_station_weather_aligned  (dbt intermediate)
  → feat_station_snapshot_5min   (dbt feature)
  → feat_station_snapshot_latest (dbt feature)
```

The full dbt intermediate layer:
- `int_station_neighbors` — K=5 neighbor graph computation
- `int_station_status_enriched` — station status joined with dimension data
- `int_station_weather_aligned` — station snapshots aligned to weather observations
- `int_station_rollups` — rolling window aggregations (15/30/60 min)

Feature tables consumed by Python:
- `feat_station_snapshot_5min` — one row per station per 5-minute bucket, full feature set
- `feat_station_snapshot_latest` — most recent snapshot per station, for online serving

### Planned Feature-Layer Contract

Formal feature generation is dbt-owned. Python training and online prediction consume the Postgres feature tables and no longer rebuild formal features in Athena/Python.

The warehouse direction is to let dbt own curated and feature-facing weather/date logic. Downstream MLOps work remains, but the warehouse contract should stay explicit.

### Reproducibility

- Re-running ingestion with the same snapshot bucket rewrites the same staging partition for that city and bucket.
- Re-running dbt rebuilds the weather dimension deterministically from warehouse staging.
- The implemented dbt feature models consume the `dim_weather` contract and do not reuse the legacy Athena weather schema.
