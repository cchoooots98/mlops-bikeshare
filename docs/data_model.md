# Data Model

## 1. Scope
This document describes the layered warehouse model for GBFS, weather, and holiday data.

## 2. Layers
### Raw
- S3 raw payloads under:
  - `raw/station_information/...`
  - `raw/station_status/...`
  - `raw/weather/...`

### Staging
- `stg_station_information`
- `stg_station_status`
- `stg_weather_current`
- `stg_weather_hourly`
- `stg_holidays`

### Transformation
- dbt builds dimensional models from staging tables.
- Weather business logic is implemented in dbt, not in ingestion code.

### Dimensions / Facts
- `dim_station`
- `dim_date`
- `dim_time`
- `dim_weather`
- `fact_station_status`

## 3. Weather Staging Tables
### stg_weather_current
- run_id (text)
- ingested_at (timestamptz)
- source_last_updated (bigint)
- city (text)
- snapshot_bucket_at (timestamptz)
- observed_at (timestamptz)
- temperature_c (double precision)
- humidity_pct (double precision)
- wind_speed_ms (double precision)
- precipitation_mm (double precision)
- weather_code (integer)
- weather_main (text)
- weather_description (text)
- source (text)

### stg_weather_hourly
- run_id (text)
- ingested_at (timestamptz)
- source_last_updated (bigint)
- city (text)
- snapshot_bucket_at (timestamptz)
- forecast_at (timestamptz)
- forecast_horizon_min (integer)
- temperature_c (double precision)
- humidity_pct (double precision)
- wind_speed_ms (double precision)
- precipitation_mm (double precision)
- precipitation_probability_pct (double precision)
- weather_code (integer)
- weather_main (text)
- weather_description (text)
- source (text)

## 4. dim_weather
`dim_weather` is built in dbt by joining `stg_weather_current` with hourly forecast rows from `stg_weather_hourly`.

Columns:
- weather_key
- city
- observed_at
- temperature_c
- humidity_pct
- wind_speed_ms
- current_precipitation_mm
- next_hour_precipitation_mm
- next_hour_precipitation_probability_pct
- rain_next_hour_flag
- next_hour_valid_at
- weather_code
- weather_main
- weather_description
- source

## 5. Relationships
- `stg_station_information.station_id -> dim_station.station_id`
- `stg_station_status.station_id -> dim_station.station_id`
- `fact_station_status.observed_at -> dim_weather.observed_at`
- `stg_holidays.holiday_date -> dim_date.date`

## 6. Weather Transformation Logic
- Ingestion writes raw current weather and raw hourly forecasts into separate staging tables.
- dbt filters hourly rows to the next 60 minutes relative to each current snapshot.
- dbt computes:
  - `next_hour_precipitation_mm`
  - `next_hour_precipitation_probability_pct`
  - `rain_next_hour_flag`
- dbt joins the summary back onto current weather to build `dim_weather`.

## 7. Star-Schema Diagram
- Mermaid source: [day3_star_schema.mmd](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/diagrams/day3_star_schema.mmd)
