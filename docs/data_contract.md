# Data Contract - Raw Layer and Warehouse Boundary

## Scope
- GBFS ingestion runs on two cadences:
  - `station_status`: every 5 minutes
  - `station_information`: daily
- Weather ingestion runs every 10 minutes.
- Holiday ingestion runs yearly or on manual replay.
- All raw timestamps are stored in UTC.
- Raw payloads are partitioned by city and `dt=YYYY-MM-DD-HH-MM`.

## Landing Paths
- `s3://<bucket>/raw/station_information/city=<city>/dt=YYYY-MM-DD-HH-MM/`
- `s3://<bucket>/raw/station_status/city=<city>/dt=YYYY-MM-DD-HH-MM/`
- `s3://<bucket>/raw/weather/city=<city>/dt=YYYY-MM-DD-HH-MM/`
- `s3://<bucket>/raw/holidays/country=<country_code>/year=<year>/dt=YYYY-MM-DD-HH-MM/`

## Required Raw Payloads
- **station_information**:
  - `data.stations[].station_id`
  - `data.stations[].name`
  - `data.stations[].capacity`
  - `data.stations[].lat`
  - `data.stations[].lon`
- **station_status**:
  - `data.stations[].station_id`
  - `data.stations[].num_bikes_available`
  - `data.stations[].num_docks_available`
  - `data.stations[].last_reported`
- **OpenWeather current block**:
  - `current.dt`
  - `current.temp`
  - `current.humidity`
  - `current.wind_speed`
  - `current.weather`
- **OpenWeather hourly block**:
  - `hourly[].dt`
  - `hourly[].temp`
  - `hourly[].humidity`
  - `hourly[].wind_speed`
  - `hourly[].pop`
  - optional `hourly[].rain.1h` / `hourly[].snow.1h`
  - `hourly[].weather`
- **Holiday API block**:
  - JSON object of `YYYY-MM-DD -> holiday_name`

## Ingestion Boundary
- Python ingestion handles:
  - API calls
  - raw payload validation
  - raw S3 writes
  - normalized staging writes into `public.stg_*`
- Python ingestion does not own:
  - `analytics.dim_weather`
  - `analytics.dim_date`
  - future curated, intermediate, or feature tables

## Warehouse Staging Contract
### `stg_station_information`
- grain: one row per `city + snapshot_bucket_at + station_id`
- lineage columns retained in staging only:
  - `run_id`
  - `ingested_at`
  - `source_last_updated`
- columns include:
  - `run_id`
  - `ingested_at`
  - `source_last_updated`
  - `city`
  - `snapshot_bucket_at`
  - `station_id`
  - `name`
  - `lat`
  - `lon`
  - `capacity`

### `stg_station_status`
- grain: one row per `city + snapshot_bucket_at + station_id`
- lineage columns retained in staging only:
  - `run_id`
  - `ingested_at`
  - `source_last_updated`
- columns include:
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

### `stg_weather_current`
- grain: one row per `city + snapshot_bucket_at`
- columns include:
  - `snapshot_bucket_at`
  - `observed_at`
  - `temperature_c`
  - `humidity_pct`
  - `wind_speed_ms`
  - `precipitation_mm`
  - `weather_code`
  - `weather_main`
  - `weather_description`

### `stg_weather_hourly`
- grain: one row per `city + snapshot_bucket_at + forecast_at`
- only includes forecast rows within the next 60 minutes of the matching `stg_weather_current.observed_at`
- columns include:
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

### `stg_holidays`
- grain: one row per `country_code + holiday_date`
- columns include:
  - `country_code`
  - `holiday_date`
  - `is_holiday`
  - `holiday_name`

## Curated Warehouse Contract
### `dim_station`
- dbt owns the station SCD2 dimension
- durable business key: `station_key = city + station_id`
- grain: one row per station version
- columns include:
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

### `fct_station_status`
- dbt owns the normalized station status fact
- grain: one row per `city + station_id + snapshot_bucket_at`
- columns include:
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
- intentionally excludes:
  - `capacity`
  - utilization fields
  - weather columns

### `dim_weather`
- dbt owns weather summarization and builds `dim_weather`
- source of truth for the current warehouse weather contract
- feature-facing columns include:
  - `temperature_c`
  - `humidity_pct`
  - `wind_speed_ms`
  - `precipitation_mm`
  - `weather_code`
  - `weather_main`
  - `hourly_temperature_c`
  - `hourly_humidity_pct`
  - `hourly_wind_speed_ms`
  - `hourly_precipitation_mm`
  - `hourly_precipitation_probability_pct`
  - `hourly_weather_code`
  - `hourly_weather_main`
- hourly enrichment rule:
  - join `stg_weather_hourly` on the same `city + run_id + snapshot_bucket_at`
  - use the row with the latest `forecast_at` as the anchor row
  - for each hourly field, backfill nulls from earlier `forecast_at` rows in the same ingest bucket

### `dim_date`
- dbt builds `dim_date` from `stg_holidays` and station-date coverage
- dbt owns:
  - `is_weekend`
  - `is_holiday`
  - `holiday_name`

## Planned Feature-Layer Contract
- The current repository still contains Athena-based feature build, training, and dashboard paths for the later MLOps stages.
- The warehouse direction is to let dbt own curated and later feature-facing weather/date logic.
- This does not mean the downstream MLOps work is removed; it means the warehouse contract should stay explicit.

Target weather feature columns:
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

Deprecated weather feature columns:
- `temp_c`
- `precip_mm`
- `wind_kph`
- `rhum_pct`
- `pres_hpa`
- `wind_dir_deg`
- `wind_gust_kph`
- `snow_mm`

## Validation
- Raw payload validation happens in ingestion code before S3 write and before staging insert.
- Invalid payloads should fail the task and prevent partial structured writes.
- Station duplicate protection is keyed by `city + snapshot_bucket_at` for both station staging tables.
- Weather duplicate protection is keyed by `city + snapshot_bucket_at` in `stg_weather_current`.
- Holiday duplicate protection is keyed by `country_code + holiday_date` within the yearly load.

## Error Handling
- Reject bad batches and log task failure in Airflow.
- Persist only valid raw payloads to `raw/...`.
- Investigate schema drift if OpenWeather stops returning `current.dt` or `hourly[]`.
- Holiday ingestion no longer creates or updates `dim_date`; dbt is responsible for rebuilding it from staging.

## Transformation Boundary
- Python ingestion handles raw ingestion and normalized staging inserts only.
- dbt handles station SCD2 logic and builds `dim_station`.
- dbt handles base station fact logic and builds `fct_station_status`.
- dbt handles weather summarization and builds `dim_weather`.
- dbt handles holiday/date logic and builds `dim_date`.
- Later machine-learning feature builds may continue to evolve, but the warehouse truth source should remain explicit and documented.

## Latency SLO
- GBFS raw landing: <= 3 minutes end-to-S3.
- Weather raw landing: <= 10 minutes from scheduled bucket.
