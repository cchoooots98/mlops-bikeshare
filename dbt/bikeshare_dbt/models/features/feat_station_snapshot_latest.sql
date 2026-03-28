{% set feature_rebuild_lookback_minutes = var('feature_rebuild_lookback_minutes', none) %}
{% if feature_rebuild_lookback_minutes is none %}
{% set feature_rebuild_lookback_minutes = (var('feature_rebuild_lookback_days', 3) | int) * 24 * 60 %}
{% else %}
{% set feature_rebuild_lookback_minutes = feature_rebuild_lookback_minutes | int %}
{% endif %}

{{ config(
    materialized='incremental',
    unique_key=['city', 'station_id'],
    incremental_strategy='delete+insert',
    on_schema_change='fail',
    post_hook=[
        "create unique index if not exists idx_feat_station_snapshot_latest_pk on {{ this }} (city, station_id)",
        "create index if not exists idx_feat_station_snapshot_latest_city_dt on {{ this }} (city, dt desc)"
    ]
) }}

with changed_station_keys as (
    {% if is_incremental() %}
    select distinct
        city,
        station_id
    from {{ ref('feat_station_snapshot_5min') }}
    where {{ feature_dt_to_utc('dt') }} >= {{ runtime_utc_expr('feature_window_end_utc') }}
        - interval '{{ feature_rebuild_lookback_minutes }} minutes'
      and {{ feature_dt_to_utc('dt') }} < {{ runtime_utc_expr('feature_window_end_utc') }}
    {% else %}
    select distinct
        city,
        station_id
    from {{ ref('feat_station_snapshot_5min') }}
    {% endif %}
),
ranked as (
    select
        f.city,
        f.dt,
        f.station_id,
        f.capacity,
        f.lat,
        f.lon,
        f.bikes,
        f.docks,
        f.minutes_since_prev_snapshot,
        f.util_bikes,
        f.util_docks,
        f.delta_bikes_5m,
        f.delta_docks_5m,
        f.roll15_net_bikes,
        f.roll30_net_bikes,
        f.roll60_net_bikes,
        f.roll15_bikes_mean,
        f.roll30_bikes_mean,
        f.roll60_bikes_mean,
        f.nbr_bikes_weighted,
        f.nbr_docks_weighted,
        f.has_neighbors_within_radius,
        f.neighbor_count_within_radius,
        f.hour,
        f.dow,
        f.is_weekend,
        f.is_holiday,
        f.temperature_c,
        f.humidity_pct,
        f.wind_speed_ms,
        f.precipitation_mm,
        f.weather_code,
        f.hourly_temperature_c,
        f.hourly_humidity_pct,
        f.hourly_wind_speed_ms,
        f.hourly_precipitation_mm,
        f.hourly_precipitation_probability_pct,
        f.hourly_weather_code,
        row_number() over (
            partition by f.city, f.station_id
            order by f.dt desc
        ) as row_num
    from {{ ref('feat_station_snapshot_5min') }} f
    inner join changed_station_keys c
        on f.city = c.city
       and f.station_id = c.station_id
)
select
    city,
    dt,
    station_id,
    capacity,
    lat,
    lon,
    bikes,
    docks,
    minutes_since_prev_snapshot,
    util_bikes,
    util_docks,
    delta_bikes_5m,
    delta_docks_5m,
    roll15_net_bikes,
    roll30_net_bikes,
    roll60_net_bikes,
    roll15_bikes_mean,
    roll30_bikes_mean,
    roll60_bikes_mean,
    nbr_bikes_weighted,
    nbr_docks_weighted,
    has_neighbors_within_radius,
    neighbor_count_within_radius,
    hour,
    dow,
    is_weekend,
    is_holiday,
    temperature_c,
    humidity_pct,
    wind_speed_ms,
    precipitation_mm,
    weather_code,
    hourly_temperature_c,
    hourly_humidity_pct,
    hourly_wind_speed_ms,
    hourly_precipitation_mm,
    hourly_precipitation_probability_pct,
    hourly_weather_code
from ranked
where row_num = 1
