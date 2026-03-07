# Feature Store Documentation

## Alignment Between Offline and Online
- Time granularity: all bike-station features are aligned on 5-minute `dt`.
- Window definitions: the same rolling windows (15/30/60 minutes) are intended for offline and online use.
- Thresholds: stockout threshold = 2 bikes/docks.
- Weather source: weather originates from OpenWeather One Call 3.0 raw payloads, lands in warehouse staging tables, and is summarized into `dim_weather` by dbt.
- Neighbor strategy: K=5 nearest neighbors within radius 0.8 km, weighted by `1/distance`.
- Ownership target: dbt will become the long-term owner of production features; Python remains the consumer of final feature tables.

## Weather Features
The curated weather dimension exposes one row per weather observation time with:

- `temperature_c`
- `humidity_pct`
- `wind_speed_ms`
- `current_precipitation_mm`
- `next_hour_precipitation_mm`
- `next_hour_precipitation_probability_pct`
- `rain_next_hour_flag`
- `weather_code`
- `weather_main`
- `weather_description`

The planned feature layer will keep only the following weather fields as first-class model features:

- `temperature_c`
- `humidity_pct`
- `wind_speed_ms`
- `current_precipitation_mm`
- `next_hour_precipitation_mm`
- `next_hour_precipitation_probability_pct`
- `rain_next_hour_flag`
- `weather_code`

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
- Business logic such as "next hour rain" is not computed in Python ingestion.
- Python weather ingestion stores only current observations and the next 60 minutes of hourly forecast rows.
- Current implemented flow:
  - raw weather JSON
  - `stg_weather_current`
  - `stg_weather_hourly`
  - dbt `dim_weather`
- Planned next flow:
  - dbt `intermediate/`
  - dbt `feat_station_snapshot_5min`
  - dbt `feat_station_snapshot_latest`

## Reproducibility
- Re-running ingestion with the same snapshot bucket rewrites the same staging partition for that city and bucket.
- Re-running dbt rebuilds the weather dimension deterministically from warehouse staging.
- Future dbt feature models will consume only the `dim_weather` contract and will not reuse the legacy Athena weather schema.
