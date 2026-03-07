# Data Contract - Raw Layer (GBFS + OpenWeather)

## Scope
- GBFS ingestion runs on two cadences:
  - `station_status`: every 5 minutes
  - `station_information`: daily
- Weather ingestion runs every 10 minutes.
- All raw timestamps are stored in UTC.
- Raw payloads are partitioned by city and `dt=YYYY-MM-DD-HH-MM`.

## Landing Paths
- `s3://<bucket>/raw/station_information/city=<city>/dt=YYYY-MM-DD-HH-MM/`
- `s3://<bucket>/raw/station_status/city=<city>/dt=YYYY-MM-DD-HH-MM/`
- `s3://<bucket>/raw/weather/city=<city>/dt=YYYY-MM-DD-HH-MM/`

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

## Warehouse Staging Contract
### stg_weather_current
- one row per city + weather snapshot bucket
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

### stg_weather_hourly
- one row per city + weather snapshot bucket + forecast timestamp
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

## Validation
- Raw payload validation happens in ingestion code before S3 write and before staging insert.
- Invalid payloads should fail the task and prevent partial structured writes.
- Duplicate protection is keyed by city plus 10-minute snapshot bucket in `stg_weather_current`.

## Error Handling
- Reject bad batches and log task failure in Airflow.
- Persist only valid raw payloads to `raw/...`.
- Investigate schema drift if OpenWeather stops returning `current.dt` or `hourly[]`.
- Holiday ingestion creates `dim_date` if it is missing before applying yearly updates.

## Transformation Boundary
- Python ingestion handles raw ingestion and normalized staging inserts only.
- dbt handles weather summarization and builds `dim_weather`.

## Latency SLO
- GBFS raw landing: <= 3 minutes end-to-S3.
- Weather raw landing: <= 10 minutes from scheduled bucket.
