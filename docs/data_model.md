# Data Model

## Scope
This document defines the dbt-first warehouse boundary for GBFS, weather, and holiday data.

## Current Implemented Layers
### Raw
- S3 raw payloads:
  - `raw/station_information/...`
  - `raw/station_status/...`
  - `raw/weather/...`
  - `raw/holidays/...`

### Python Ingestion to `public`
- `stg_station_information`
- `stg_station_status`
- `stg_weather_current`
- `stg_weather_hourly`
- `stg_holidays`

### dbt Curated in `analytics`
- `dim_station`
- `dim_weather`
- `dim_date`

`dim_date` is now owned by dbt. Holiday ingestion stops at `stg_holidays`.

## Planned dbt Layers
These layers are part of the target architecture but are not implemented in this phase.

### Curated / Fact
- `dim_time`
- `fct_station_status`

### Intermediate
- `int_station_neighbors`
- `int_station_status_enriched`
- `int_station_weather_aligned`
- `int_station_rollups`

### Features
- `feat_station_snapshot_5min`
- `feat_station_snapshot_latest`

## Station Model Contract
Station data should retain `city` from raw landing through staging, dimensions, and future facts. In production terms, `station_id` by itself is not a safe long-term business key once the warehouse may hold more than one city.

### `stg_station_information`
- `run_id`
- `ingested_at`
- `source_last_updated`
- `city`
- `station_id`
- `station_name`
- `latitude`
- `longitude`
- `capacity`

### `stg_station_status`
- `run_id`
- `ingested_at`
- `source_last_updated`
- `city`
- `station_id`
- `last_reported_at`
- `num_bikes_available`
- `num_docks_available`
- `is_renting`
- `is_returning`

### `dim_station`
- `station_key`
- `city`
- `station_id`
- `station_name`
- `latitude`
- `longitude`
- `capacity`

### Future `fct_station_status`
- should retain `city`
- should join station via `station_key`
- should keep uniqueness scoped by `city + observed_at + station_id`

## Weather Model Contract
`dim_weather` is the single source of truth for weather features. It is built from `stg_weather_current` and `stg_weather_hourly`.
For each current-weather row, dbt joins hourly rows from the same `city + run_id + snapshot_bucket_at`, takes the latest `forecast_at` as the reference hourly row, and backfills null hourly fields from earlier `forecast_at` rows in that same ingest bucket.

### `stg_weather_current`
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

### `stg_weather_hourly`
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

### `dim_weather`
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

## Weather Feature Direction
The planned dbt feature layer will consume the following weather columns from `dim_weather`:

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

The following legacy fields are no longer part of the target feature contract:

- `temp_c`
- `precip_mm`
- `wind_kph`
- `rhum_pct`
- `pres_hpa`
- `wind_dir_deg`
- `wind_gust_kph`
- `snow_mm`

## Holiday and Date Logic
- Holiday raw API responses are stored in S3 under `raw/holidays/country=.../year=.../dt=.../`.
- `stg_holidays` stores one row per holiday date for a country/year load.
- dbt builds `dim_date` from `stg_holidays` and available station-status date bounds.
- Weekend and holiday attributes belong to `analytics.dim_date`, not to Python ingestion.

## Diagram
- Mermaid source: [day3_star_schema.mmd](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/diagrams/day3_star_schema.mmd)
