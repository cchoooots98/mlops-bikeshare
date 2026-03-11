{{ config(
    materialized='view'
) }}

with src as (
    select * from {{ source('raw_staging', 'stg_weather_current') }}
),
ranked as (
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
        temperature_c::double precision as temperature_c,
        humidity_pct::double precision as humidity_pct,
        wind_speed_ms::double precision as wind_speed_ms,
        precipitation_mm::double precision as precipitation_mm,
        weather_code::integer as weather_code,
        weather_main::text as weather_main,
        weather_description::text as weather_description,
        row_number() over (
            partition by city::text, observed_at::timestamptz
            order by
                source_last_updated::bigint desc,
                ingested_at::timestamptz desc,
                run_id::text desc
        ) as row_num
    from src
)
select
    run_id,
    city,
    source,
    source_last_updated,
    ingested_at_utc,
    ingested_at_paris,
    snapshot_bucket_at_utc,
    snapshot_bucket_at_paris,
    observed_at_utc,
    observed_at_paris,
    temperature_c,
    humidity_pct,
    wind_speed_ms,
    precipitation_mm,
    weather_code,
    weather_main,
    weather_description,
    concat(city, '|', {{ utc_ts_key('observed_at_utc') }}) as weather_current_pk
from ranked
where row_num = 1
