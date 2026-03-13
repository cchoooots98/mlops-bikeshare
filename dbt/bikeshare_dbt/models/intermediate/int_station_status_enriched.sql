{% set weather_asof_tolerance_minutes = var('station_status_weather_asof_tolerance_minutes', 30) | int %}
{% set enrich_rebuild_lookback_minutes = var('enrich_rebuild_lookback_minutes', 30) | int %}
{% set enrich_history_buffer_minutes = var('enrich_history_buffer_minutes', 60) | int %}
{% set existing_relation = adapter.get_relation(database=this.database, schema=this.schema, identifier=this.identifier) %}
{% set run_recent_window_overwrite = existing_relation is not none and not flags.FULL_REFRESH %}
{% set recent_window_overwrite_sql %}
{% if run_recent_window_overwrite %}
delete from {{ this }}
where snapshot_bucket_at_utc >= {{ runtime_utc_expr('hotpath_window_end_utc') }}
    - interval '{{ enrich_rebuild_lookback_minutes }} minutes'
  and snapshot_bucket_at_utc < {{ runtime_utc_expr('hotpath_window_end_utc') }}
{% else %}
select 1
{% endif %}
{% endset %}

{{ config(
    materialized='incremental',
    unique_key='fact_station_status_key',
    incremental_strategy='delete+insert',
    on_schema_change='fail',
    pre_hook=[recent_window_overwrite_sql],
    post_hook=[
        "create index if not exists idx_int_station_status_enriched_station_ts on {{ this }} (city, station_id, snapshot_bucket_at_utc)"
    ]
) }}

with window_bounds as (
    select
        {{ runtime_utc_expr('hotpath_window_end_utc') }} as rebuild_to_utc,
        {{ runtime_utc_expr('hotpath_window_end_utc') }}
            - interval '{{ enrich_rebuild_lookback_minutes }} minutes' as rebuild_from_utc,
        {{ runtime_utc_expr('hotpath_window_end_utc') }}
            - interval '{{ enrich_rebuild_lookback_minutes + enrich_history_buffer_minutes }} minutes' as source_from_utc
),
fact_base as (
    select f.*
    from {{ ref('fct_station_status') }} f
    {% if is_incremental() %}
    cross join window_bounds wb
    where f.snapshot_bucket_at_utc >= wb.source_from_utc
      and f.snapshot_bucket_at_utc < wb.rebuild_to_utc
    {% endif %}
),
station_dim_intersection as (
    select
        f.fact_station_status_key,
        f.city,
        f.station_id,
        f.station_key,
        f.station_version_key,
        f.snapshot_bucket_at_utc,
        f.snapshot_bucket_at_paris,
        f.date_id,
        f.time_id,
        f.last_reported_at_utc,
        f.last_reported_at_paris,
        f.num_bikes_available,
        f.num_docks_available,
        f.is_renting,
        f.is_returning,
        s.capacity,
        s.latitude,
        s.longitude
    from fact_base f
    inner join {{ ref('dim_station') }} s
        on f.station_version_key = s.station_version_key
    where s.capacity is not null
      and {{ station_inventory_within_limit_expr('f.num_bikes_available', 'f.num_docks_available', 's.capacity') }}
),
station_date_time_enriched as (
    select
        f.fact_station_status_key,
        f.city,
        f.station_id,
        f.station_key,
        f.station_version_key,
        f.snapshot_bucket_at_utc,
        f.snapshot_bucket_at_paris,
        f.date_id,
        f.time_id,
        f.last_reported_at_utc,
        f.last_reported_at_paris,
        f.num_bikes_available,
        f.num_docks_available,
        f.is_renting,
        f.is_returning,
        f.capacity,
        f.latitude,
        f.longitude,
        d.date,
        d.day_of_week::integer as day_of_week,
        d.is_weekend,
        d.is_holiday,
        t.hour,
        t.minute,
        t.minute_of_day
    from station_dim_intersection f
    left join {{ ref('dim_date') }} d
        on f.date_id = d.date_id
    left join {{ ref('dim_time') }} t
        on f.time_id = t.time_id
),
weather_enriched as (
    select
        e.*,
        w.weather_key,
        w.weather_observed_at_utc,
        w.temperature_c,
        w.humidity_pct,
        w.wind_speed_ms,
        w.precipitation_mm,
        w.weather_code,
        w.hourly_forecast_at,
        w.hourly_temperature_c,
        w.hourly_humidity_pct,
        w.hourly_wind_speed_ms,
        w.hourly_precipitation_mm,
        w.hourly_precipitation_probability_pct,
        w.hourly_weather_code,
        w.weather_asof_lag_minutes,
        coalesce(w.has_weather_context, 0)::smallint as has_weather_context
    from station_date_time_enriched e
    left join lateral (
        select
            dw.weather_key,
            dw.observed_at as weather_observed_at_utc,
            dw.temperature_c,
            dw.humidity_pct,
            dw.wind_speed_ms,
            dw.precipitation_mm,
            dw.weather_code,
            dw.hourly_forecast_at,
            dw.hourly_temperature_c,
            dw.hourly_humidity_pct,
            dw.hourly_wind_speed_ms,
            dw.hourly_precipitation_mm,
            dw.hourly_precipitation_probability_pct,
            dw.hourly_weather_code,
            extract(epoch from (e.snapshot_bucket_at_utc - dw.observed_at)) / 60.0 as weather_asof_lag_minutes,
            1::smallint as has_weather_context
        from {{ ref('dim_weather') }} dw
        where dw.city = e.city
          and dw.observed_at <= e.snapshot_bucket_at_utc
          and dw.observed_at >= e.snapshot_bucket_at_utc - interval '{{ weather_asof_tolerance_minutes }} minutes'
        order by dw.observed_at desc
        limit 1
    ) w
        on true
),
temporal_helpers as (
    select
        *,
        lag(snapshot_bucket_at_utc) over (
            partition by city, station_id
            order by snapshot_bucket_at_utc
        ) as prev_snapshot_bucket_at_utc,
        lag(num_bikes_available) over (
            partition by city, station_id
            order by snapshot_bucket_at_utc
        ) as prev_num_bikes_available,
        lag(num_docks_available) over (
            partition by city, station_id
            order by snapshot_bucket_at_utc
        ) as prev_num_docks_available
    from weather_enriched
)
select
    fact_station_status_key,
    city,
    station_id,
    station_key,
    station_version_key,
    snapshot_bucket_at_utc,
    snapshot_bucket_at_paris,
    date_id,
    time_id,
    last_reported_at_utc,
    last_reported_at_paris,
    num_bikes_available,
    num_docks_available,
    is_renting,
    is_returning,
    capacity,
    latitude,
    longitude,
    date,
    day_of_week,
    is_weekend,
    is_holiday,
    hour,
    minute,
    minute_of_day,
    weather_key,
    weather_observed_at_utc,
    temperature_c,
    humidity_pct,
    wind_speed_ms,
    precipitation_mm,
    weather_code,
    hourly_forecast_at,
    hourly_temperature_c,
    hourly_humidity_pct,
    hourly_wind_speed_ms,
    hourly_precipitation_mm,
    hourly_precipitation_probability_pct,
    hourly_weather_code,
    weather_asof_lag_minutes::double precision as weather_asof_lag_minutes,
    has_weather_context,
    prev_snapshot_bucket_at_utc,
    (
        extract(epoch from (snapshot_bucket_at_utc - prev_snapshot_bucket_at_utc)) / 60.0
    )::double precision as minutes_since_prev_snapshot,
    prev_num_bikes_available,
    prev_num_docks_available
from temporal_helpers
{% if is_incremental() %}
where snapshot_bucket_at_utc >= (select rebuild_from_utc from window_bounds)
  and snapshot_bucket_at_utc < (select rebuild_to_utc from window_bounds)
{% endif %}
