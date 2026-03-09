{{ config(
    materialized='view'
) }}

with src as (
    select * from {{ source('raw_staging', 'stg_weather_hourly') }}
)
select
    run_id::text as run_id,
    city::text as city,
    source::text as source,
    source_last_updated::bigint as source_last_updated,
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
    concat(
        city::text,
        '|',
        to_char(snapshot_bucket_at::timestamptz, 'YYYY-MM-DD HH24:MI:SSOF'),
        '|',
        to_char(forecast_at::timestamptz, 'YYYY-MM-DD HH24:MI:SSOF')
    ) as weather_hourly_pk
from src
