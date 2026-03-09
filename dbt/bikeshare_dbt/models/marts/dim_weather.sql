{{ config(
    materialized='incremental',
    unique_key='weather_key',
    on_schema_change='sync_all_columns'
) }}

with current_weather as (
    select *
    from {{ ref('stg_weather_current') }}
    {% if is_incremental() %}
    where observed_at_utc > (select coalesce(max(observed_at), '1900-01-01'::timestamptz) from {{ this }})
    {% endif %}
),
hourly_merged as (
    select
        c.weather_current_pk,
        c.city,
        c.run_id,
        c.snapshot_bucket_at_utc,
        c.observed_at_utc,
        max(h.forecast_at_utc) as hourly_forecast_at,
        (array_agg(h.temperature_c order by h.forecast_at_utc desc)
            filter (where h.temperature_c is not null))[1] as hourly_temperature_c,
        (array_agg(h.humidity_pct order by h.forecast_at_utc desc)
            filter (where h.humidity_pct is not null))[1] as hourly_humidity_pct,
        (array_agg(h.wind_speed_ms order by h.forecast_at_utc desc)
            filter (where h.wind_speed_ms is not null))[1] as hourly_wind_speed_ms,
        (array_agg(h.precipitation_mm order by h.forecast_at_utc desc)
            filter (where h.precipitation_mm is not null))[1] as hourly_precipitation_mm,
        (array_agg(h.precipitation_probability_pct order by h.forecast_at_utc desc)
            filter (where h.precipitation_probability_pct is not null))[1] as hourly_precipitation_probability_pct,
        (array_agg(h.weather_code order by h.forecast_at_utc desc)
            filter (where h.weather_code is not null))[1] as hourly_weather_code,
        (array_agg(h.weather_main order by h.forecast_at_utc desc)
            filter (where h.weather_main is not null))[1] as hourly_weather_main
    from current_weather c
    left join {{ ref('stg_weather_hourly') }} h
        on c.city = h.city
       and c.run_id = h.run_id
       and c.snapshot_bucket_at_utc = h.snapshot_bucket_at_utc
       and c.observed_at_utc = h.observed_at_utc
    group by
        c.weather_current_pk,
        c.city,
        c.run_id,
        c.snapshot_bucket_at_utc,
        c.observed_at_utc
)
select
    concat(
        c.city,
        '|',
        to_char(c.observed_at_utc, 'YYYY-MM-DD HH24:MI:SSOF')
    ) as weather_key,
    c.city,
    c.observed_at_utc as observed_at,
    c.temperature_c,
    c.humidity_pct,
    c.wind_speed_ms,
    c.precipitation_mm,
    c.weather_code,
    c.weather_main,
    c.weather_description,
    h.hourly_forecast_at,
    h.hourly_temperature_c,
    h.hourly_humidity_pct,
    h.hourly_wind_speed_ms,
    h.hourly_precipitation_mm,
    h.hourly_precipitation_probability_pct,
    h.hourly_weather_code,
    h.hourly_weather_main,
    c.source,
    c.snapshot_bucket_at_utc
from current_weather c
left join hourly_merged h
    on c.weather_current_pk = h.weather_current_pk
