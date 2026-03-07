# Feature Store Documentation

## 1. Alignment Between Offline and Online
- **Time granularity:** all bike station features are aligned on 5-minute `dt`.
- **Window definitions:** the same rolling windows (15/30/60 minutes) are used offline and online.
- **Thresholds:** stockout threshold = 2 bikes/docks.
- **Weather source:** weather originates from OpenWeather One Call 3.0 raw payloads, lands in warehouse staging tables, and is summarized into `dim_weather` by dbt.
- **Neighbor strategy:** K=5 nearest neighbors within radius 0.8 km, weighted by `1/distance`.

## 2. Weather Features
The curated weather dimension is designed to expose one row per weather observation time with:

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

## 3. Design Notes
- Raw weather JSON remains in S3 for replay and auditing.
- Business logic such as "next hour rain" is not computed in Python ingestion.
- Python weather ingestion stores only current observations and the next 60 minutes of hourly forecast rows.
- The warehouse flow is:
  - raw weather JSON
  - `stg_weather_current`
  - `stg_weather_hourly`
  - dbt `dim_weather`

## 4. Reproducibility
- Re-running ingestion with the same snapshot bucket rewrites the same staging partition for that city and bucket.
- Re-running dbt rebuilds the weather dimension deterministically from warehouse staging.

