{{ config(
    materialized='incremental',
    unique_key='weather_key'
) }}

with current_weather as (
    select *
    from {{ ref('stg_weather_current') }}
    {% if is_incremental() %}
    where observed_at_utc > (select coalesce(max(observed_at), '1900-01-01'::timestamptz) from {{ this }})
    {% endif %}
),
hourly_candidates as (
    select
        c.city,
        c.snapshot_bucket_at_utc,
        c.observed_at_utc,
        h.forecast_at_utc,
        h.precipitation_mm,
        h.precipitation_probability_pct,
        row_number() over (
            partition by c.city, c.snapshot_bucket_at_utc
            order by h.forecast_at_utc
        ) as rn
    from current_weather c
    left join {{ ref('stg_weather_hourly') }} h
        on c.city = h.city
       and c.snapshot_bucket_at_utc = h.snapshot_bucket_at_utc
       and h.forecast_at_utc > c.observed_at_utc
       and h.forecast_at_utc <= c.observed_at_utc + interval '1 hour'
),
hourly_summary as (
    select
        city,
        snapshot_bucket_at_utc,
        sum(coalesce(precipitation_mm, 0.0)) as next_hour_precipitation_mm,
        max(case when rn = 1 then precipitation_probability_pct end) as next_hour_precipitation_probability_pct
    from hourly_candidates
    group by city, snapshot_bucket_at_utc
),
earliest_forecast as (
    select
        city,
        snapshot_bucket_at_utc,
        forecast_at_utc as next_hour_forecast_at_utc
    from hourly_candidates
    where rn = 1
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
    c.precipitation_mm as current_precipitation_mm,
    coalesce(s.next_hour_precipitation_mm, 0.0) as next_hour_precipitation_mm,
    s.next_hour_precipitation_probability_pct,
    case
        when coalesce(s.next_hour_precipitation_mm, 0.0) > 0
          or coalesce(s.next_hour_precipitation_probability_pct, 0.0) >= 50
        then true
        else false
    end as rain_next_hour_flag,
    e.next_hour_forecast_at_utc as next_hour_valid_at,
    c.weather_code,
    c.weather_main,
    c.weather_description,
    c.source,
    c.snapshot_bucket_at_utc
from current_weather c
left join hourly_summary s
    on c.city = s.city
   and c.snapshot_bucket_at_utc = s.snapshot_bucket_at_utc
left join earliest_forecast e
    on c.city = e.city
   and c.snapshot_bucket_at_utc = e.snapshot_bucket_at_utc
