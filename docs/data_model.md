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
- `dim_weather`
- `dim_date`

`dim_date` is now owned by dbt. Holiday ingestion stops at `stg_holidays`.

## Planned dbt Layers
These layers are part of the target architecture but are not implemented in this phase.

### Curated / Fact
- `dim_station`
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

## Weather Model Contract
`dim_weather` is the single source of truth for weather features. It is built from `stg_weather_current` and `stg_weather_hourly`.

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
- `current_precipitation_mm`
- `next_hour_precipitation_mm`
- `next_hour_precipitation_probability_pct`
- `rain_next_hour_flag`
- `next_hour_valid_at`
- `weather_code`
- `weather_main`
- `weather_description`
- `source`
- `snapshot_bucket_at_utc`

## Weather Feature Direction
The planned dbt feature layer will consume only the following weather feature columns:

- `temperature_c`
- `humidity_pct`
- `wind_speed_ms`
- `current_precipitation_mm`
- `next_hour_precipitation_mm`
- `next_hour_precipitation_probability_pct`
- `rain_next_hour_flag`
- `weather_code`

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
