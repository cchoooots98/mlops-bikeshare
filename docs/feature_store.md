# Feature Store Documentation

## Alignment Between Offline and Online
- Time granularity: all bike-station features are aligned on 5-minute `dt`.
- Window definitions: the same rolling windows (15/30/60 minutes) are intended for offline and online use.
- Thresholds: stockout threshold = 2 bikes/docks.
- Weather source: weather originates from OpenWeather One Call 3.0 raw payloads, lands in warehouse staging tables, and is summarized into `dim_weather` by dbt.
- Neighbor strategy: K=5 nearest neighbors within radius 0.8 km, weighted by `1/distance`.
- Ownership: dbt is now the formal producer of warehouse feature tables; Python remains the consumer of final feature tables.

## Weather Features
The curated weather dimension exposes one row per weather observation time with:

- `temperature_c`
- `humidity_pct`
- `wind_speed_ms`
- `precipitation_mm`
- `weather_code`

- `weather_description`
- `hourly_forecast_at`
- `hourly_temperature_c`
- `hourly_humidity_pct`
- `hourly_wind_speed_ms`
- `hourly_precipitation_mm`
- `hourly_precipitation_probability_pct`
- `hourly_weather_code`


The implemented feature layer keeps only the following weather fields as first-class model features:

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

Legacy weather feature fields to remove in the next feature migration:

- `temp_c`
- `precip_mm`
- `wind_kph`
- `rhum_pct`
- `pres_hpa`
- `wind_dir_deg`
- `wind_gust_kph`
- `snow_mm`

## Current and Planned Layers
- Raw weather JSON remains in S3 for replay and auditing.
- Python ingestion does not collapse hourly forecast rows into a single weather contract row.
- Python weather ingestion stores only current observations and the next 60 minutes of hourly forecast rows.
- dbt merges those hourly rows by staying within the same `city + run_id + snapshot_bucket_at` bucket, using the latest `forecast_at` as the anchor hourly row, and backfilling null hourly fields from earlier forecast rows.
- Current implemented flow:
  - raw weather JSON
  - `stg_weather_current`
  - `stg_weather_hourly`
  - dbt `dim_weather`
- Implemented current feature flow:
  - dbt `intermediate/`
  - dbt `feat_station_snapshot_5min`
  - dbt `feat_station_snapshot_latest`

## Reproducibility
- Re-running ingestion with the same snapshot bucket rewrites the same staging partition for that city and bucket.
- Re-running dbt rebuilds the weather dimension deterministically from warehouse staging.
- The implemented dbt feature models consume the `dim_weather` contract and do not reuse the legacy Athena weather schema.
