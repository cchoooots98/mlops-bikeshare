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
    union
    select distinct
        city,
        station_id
    from {{ this }}
    where {{ feature_dt_to_utc('dt') }} >= {{ runtime_utc_expr('feature_window_end_utc') }}
        - interval '{{ feature_rebuild_lookback_minutes }} minutes'
    {% else %}
    select distinct
        city,
        station_id
    from {{ ref('feat_station_snapshot_5min') }}
    {% endif %}
),
historical_fallback as (
    {% if is_incremental() %}
    select
        fallback.city,
        fallback.dt,
        fallback.station_id,
        fallback.capacity,
        fallback.lat,
        fallback.lon,
        fallback.bikes,
        fallback.docks,
        fallback.minutes_since_prev_snapshot,
        fallback.util_bikes,
        fallback.util_docks,
        fallback.delta_bikes_5m,
        fallback.delta_docks_5m,
        fallback.roll15_net_bikes,
        fallback.roll30_net_bikes,
        fallback.roll60_net_bikes,
        fallback.roll15_bikes_mean,
        fallback.roll30_bikes_mean,
        fallback.roll60_bikes_mean,
        fallback.nbr_bikes_weighted,
        fallback.nbr_docks_weighted,
        fallback.has_neighbors_within_radius,
        fallback.neighbor_count_within_radius,
        fallback.hour,
        fallback.dow,
        fallback.is_weekend,
        fallback.is_holiday,
        fallback.temperature_c,
        fallback.humidity_pct,
        fallback.wind_speed_ms,
        fallback.precipitation_mm,
        fallback.weather_code,
        fallback.hourly_temperature_c,
        fallback.hourly_humidity_pct,
        fallback.hourly_wind_speed_ms,
        fallback.hourly_precipitation_mm,
        fallback.hourly_precipitation_probability_pct,
        fallback.hourly_weather_code
    from changed_station_keys c
    inner join lateral (
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
            f.hourly_weather_code
        from {{ ref('feat_station_snapshot_5min') }} f
        where f.city = c.city
          and f.station_id = c.station_id
          and {{ feature_dt_to_utc('f.dt') }} < {{ runtime_utc_expr('feature_window_end_utc') }}
              - interval '{{ feature_rebuild_lookback_minutes }} minutes'
        order by f.dt desc
        limit 1
    ) fallback
        on true
    {% else %}
    select
        cast(null as text) as city,
        cast(null as text) as dt,
        cast(null as text) as station_id,
        cast(null as integer) as capacity,
        cast(null as double precision) as lat,
        cast(null as double precision) as lon,
        cast(null as integer) as bikes,
        cast(null as integer) as docks,
        cast(null as double precision) as minutes_since_prev_snapshot,
        cast(null as double precision) as util_bikes,
        cast(null as double precision) as util_docks,
        cast(null as double precision) as delta_bikes_5m,
        cast(null as double precision) as delta_docks_5m,
        cast(null as double precision) as roll15_net_bikes,
        cast(null as double precision) as roll30_net_bikes,
        cast(null as double precision) as roll60_net_bikes,
        cast(null as double precision) as roll15_bikes_mean,
        cast(null as double precision) as roll30_bikes_mean,
        cast(null as double precision) as roll60_bikes_mean,
        cast(null as double precision) as nbr_bikes_weighted,
        cast(null as double precision) as nbr_docks_weighted,
        cast(null as integer) as has_neighbors_within_radius,
        cast(null as integer) as neighbor_count_within_radius,
        cast(null as integer) as hour,
        cast(null as integer) as dow,
        cast(null as integer) as is_weekend,
        cast(null as integer) as is_holiday,
        cast(null as double precision) as temperature_c,
        cast(null as double precision) as humidity_pct,
        cast(null as double precision) as wind_speed_ms,
        cast(null as double precision) as precipitation_mm,
        cast(null as integer) as weather_code,
        cast(null as double precision) as hourly_temperature_c,
        cast(null as double precision) as hourly_humidity_pct,
        cast(null as double precision) as hourly_wind_speed_ms,
        cast(null as double precision) as hourly_precipitation_mm,
        cast(null as double precision) as hourly_precipitation_probability_pct,
        cast(null as integer) as hourly_weather_code
    where false
    {% endif %}
),
recent_candidates as (
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
        f.hourly_weather_code
    from {{ ref('feat_station_snapshot_5min') }} f
    inner join changed_station_keys c
        on f.city = c.city
       and f.station_id = c.station_id
    {% if is_incremental() %}
    where {{ feature_dt_to_utc('f.dt') }} >= {{ runtime_utc_expr('feature_window_end_utc') }}
        - interval '{{ feature_rebuild_lookback_minutes }} minutes'
      and {{ feature_dt_to_utc('f.dt') }} < {{ runtime_utc_expr('feature_window_end_utc') }}
    {% endif %}
),
candidates as (
    select * from recent_candidates
    union all
    select * from historical_fallback
),
ranked as (
    select
        *,
        row_number() over (
            partition by city, station_id
            order by dt desc
        ) as row_num
    from candidates
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
