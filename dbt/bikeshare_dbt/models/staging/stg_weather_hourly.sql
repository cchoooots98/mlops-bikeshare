{{ config(
    materialized='view'
) }}

with src as (
    select * from {{ source('raw_staging', 'stg_weather_hourly') }}
),
ranked as (
    select
        run_id::text as run_id,
        city::text as city,
        source::text as source,
        ingested_at::timestamptz as ingested_at_utc,
        (ingested_at::timestamptz at time zone 'Europe/Paris')::timestamp as ingested_at_paris,
        snapshot_bucket_at::timestamptz as snapshot_bucket_at_utc,
        (snapshot_bucket_at::timestamptz at time zone 'Europe/Paris')::timestamp as snapshot_bucket_at_paris,
        observed_at::timestamptz as observed_at_utc,
        (observed_at::timestamptz at time zone 'Europe/Paris')::timestamp as observed_at_paris,
        forecast_at::timestamptz as forecast_at_utc,
        (forecast_at::timestamptz at time zone 'Europe/Paris')::timestamp as forecast_at_paris,
        forecast_horizon_min::integer as forecast_horizon_min,
        temperature_c::double precision as temperature_c,
        humidity_pct::double precision as humidity_pct,
        wind_speed_ms::double precision as wind_speed_ms,
        precipitation_mm::double precision as precipitation_mm,
        precipitation_probability_pct::double precision as precipitation_probability_pct,
        weather_code::integer as weather_code,
        weather_main::text as weather_main,
        weather_description::text as weather_description,
        row_number() over (
            partition by
                city::text,
                snapshot_bucket_at::timestamptz,
                forecast_at::timestamptz
            order by
                ingested_at::timestamptz desc,
                run_id::text desc
        ) as row_num
    from src
)
select
    run_id,
    city,
    source,
    ingested_at_utc,
    ingested_at_paris,
    snapshot_bucket_at_utc,
    snapshot_bucket_at_paris,
    observed_at_utc,
    observed_at_paris,
    forecast_at_utc,
    forecast_at_paris,
    forecast_horizon_min,
    temperature_c,
    humidity_pct,
    wind_speed_ms,
    precipitation_mm,
    precipitation_probability_pct,
    weather_code,
    weather_main,
    weather_description,
    concat(
        city,
        '|',
        {{ utc_ts_key('snapshot_bucket_at_utc') }},
        '|',
        {{ utc_ts_key('forecast_at_utc') }}
    ) as weather_hourly_pk
from ranked
where row_num = 1
  and observed_at_utc is not null
  and forecast_at_utc is not null
  and forecast_horizon_min between 0 and 60
  and forecast_at_utc > observed_at_utc
  and forecast_at_utc <= observed_at_utc + interval '60 minutes'
  and (humidity_pct is null or humidity_pct between 0 and 100)
  and (wind_speed_ms is null or wind_speed_ms >= 0)
  and (precipitation_mm is null or precipitation_mm >= 0)
  and (
      precipitation_probability_pct is null
      or precipitation_probability_pct between 0 and 100
  )
