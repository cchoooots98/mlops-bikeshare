{% set stockout_threshold = var('stockout_threshold', 2) | int %}
{% set snapshot_step_minutes = var('feature_snapshot_step_minutes', 5) | int %}
{% set label_horizon_minutes = var('feature_label_horizon_minutes', 30) | int %}
{% set max_roll_window_minutes = var('feature_max_roll_window_minutes', 60) | int %}
{% set feature_rebuild_lookback_minutes = var('feature_rebuild_lookback_minutes', none) %}
{% if feature_rebuild_lookback_minutes is none %}
{% set feature_rebuild_lookback_minutes = (var('feature_rebuild_lookback_days', 3) | int) * 24 * 60 %}
{% else %}
{% set feature_rebuild_lookback_minutes = feature_rebuild_lookback_minutes | int %}
{% endif %}
{% set expected_future_snapshots = label_horizon_minutes // snapshot_step_minutes %}
{% set existing_relation = adapter.get_relation(database=this.database, schema=this.schema, identifier=this.identifier) %}
{% set run_recent_window_overwrite = existing_relation is not none and not flags.FULL_REFRESH %}
{% set recent_window_overwrite_sql %}
{% if run_recent_window_overwrite %}
delete from {{ this }}
where {{ feature_dt_to_utc('dt') }} >= {{ runtime_utc_expr('feature_window_end_utc') }}
    - interval '{{ feature_rebuild_lookback_minutes }} minutes'
  and {{ feature_dt_to_utc('dt') }} < {{ runtime_utc_expr('feature_window_end_utc') }}
{% else %}
select 1
{% endif %}
{% endset %}

{{ config(
    materialized='incremental',
    unique_key=['city', 'station_id', 'dt'],
    incremental_strategy='delete+insert',
    on_schema_change='fail',
    pre_hook=[recent_window_overwrite_sql],
    post_hook=[
        "create unique index if not exists idx_feat_station_snapshot_5min_pk on {{ this }} (city, station_id, dt)",
        "create index if not exists idx_feat_station_snapshot_5min_station_dt on {{ this }} (city, station_id, dt desc)"
    ]
) }}

with window_bounds as (
    select
        {{ runtime_utc_expr('feature_window_end_utc') }} as rebuild_to_utc,
        {{ runtime_utc_expr('feature_window_end_utc') }}
            - interval '{{ feature_rebuild_lookback_minutes }} minutes' as rebuild_from_utc,
        {{ runtime_utc_expr('feature_window_end_utc') }}
            - interval '{{ feature_rebuild_lookback_minutes + max_roll_window_minutes }} minutes' as source_from_utc,
        {{ runtime_utc_expr('feature_window_end_utc') }}
            + interval '{{ label_horizon_minutes }} minutes' as source_to_utc
),
base_source as (
    select
        e.city,
        e.station_id,
        e.station_key,
        e.snapshot_bucket_at_utc,
        e.capacity,
        e.latitude as lat,
        e.longitude as lon,
        e.num_bikes_available as bikes,
        e.num_docks_available as docks,
        e.hour,
        e.day_of_week as dow,
        e.is_weekend,
        e.is_holiday,
        e.temperature_c,
        e.humidity_pct,
        e.wind_speed_ms,
        e.precipitation_mm,
        e.weather_code,
        e.hourly_temperature_c,
        e.hourly_humidity_pct,
        e.hourly_wind_speed_ms,
        e.hourly_precipitation_mm,
        e.hourly_precipitation_probability_pct,
        e.hourly_weather_code,
        e.minutes_since_prev_snapshot,
        e.prev_num_bikes_available,
        e.prev_num_docks_available
    from {{ ref('int_station_status_enriched') }} e
    cross join window_bounds wb
    {% if is_incremental() %}
    where e.snapshot_bucket_at_utc >= wb.source_from_utc
      and e.snapshot_bucket_at_utc < wb.source_to_utc
    {% endif %}
),
station_features as (
    select
        city,
        station_id,
        station_key,
        snapshot_bucket_at_utc,
        {{ feature_dt_from_utc('snapshot_bucket_at_utc') }} as dt,
        capacity,
        lat,
        lon,
        bikes,
        docks,
        coalesce(minutes_since_prev_snapshot, 0.0)::double precision as minutes_since_prev_snapshot,
        greatest(
            0.0,
            coalesce(bikes::double precision / nullif(capacity::double precision, 0.0), 0.0)
        ) as util_bikes,
        greatest(
            0.0,
            coalesce(docks::double precision / nullif(capacity::double precision, 0.0), 0.0)
        ) as util_docks,
        case
            when coalesce(minutes_since_prev_snapshot, 0.0) > 0.0
                then coalesce((bikes - prev_num_bikes_available)::double precision, 0.0)
                    / minutes_since_prev_snapshot::double precision
                    * {{ snapshot_step_minutes }}::double precision
            else 0.0
        end as delta_bikes_5m,
        case
            when coalesce(minutes_since_prev_snapshot, 0.0) > 0.0
                then coalesce((docks - prev_num_docks_available)::double precision, 0.0)
                    / minutes_since_prev_snapshot::double precision
                    * {{ snapshot_step_minutes }}::double precision
            else 0.0
        end as delta_docks_5m,
        hour::integer as hour,
        dow::integer as dow,
        coalesce(is_weekend, false)::integer as is_weekend,
        coalesce(is_holiday, false)::integer as is_holiday,
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
    from base_source
),
station_features_windowed as (
    select
        sf.*,
        sum(delta_bikes_5m) over (
            partition by city, station_id
            order by snapshot_bucket_at_utc
            range between interval '15 minutes' preceding and current row
        ) as roll15_net_bikes,
        sum(delta_bikes_5m) over (
            partition by city, station_id
            order by snapshot_bucket_at_utc
            range between interval '30 minutes' preceding and current row
        ) as roll30_net_bikes,
        sum(delta_bikes_5m) over (
            partition by city, station_id
            order by snapshot_bucket_at_utc
            range between interval '{{ max_roll_window_minutes }} minutes' preceding and current row
        ) as roll60_net_bikes,
        avg(bikes::double precision) over (
            partition by city, station_id
            order by snapshot_bucket_at_utc
            range between interval '15 minutes' preceding and current row
        ) as roll15_bikes_mean,
        avg(bikes::double precision) over (
            partition by city, station_id
            order by snapshot_bucket_at_utc
            range between interval '30 minutes' preceding and current row
        ) as roll30_bikes_mean,
        avg(bikes::double precision) over (
            partition by city, station_id
            order by snapshot_bucket_at_utc
            range between interval '{{ max_roll_window_minutes }} minutes' preceding and current row
        ) as roll60_bikes_mean,
        count(*) over (
            partition by city, station_id
            order by snapshot_bucket_at_utc
            range between interval '{{ snapshot_step_minutes }} minutes' following
            and interval '{{ label_horizon_minutes }} minutes' following
        ) as future_snapshot_count,
        min(bikes) over (
            partition by city, station_id
            order by snapshot_bucket_at_utc
            range between interval '{{ snapshot_step_minutes }} minutes' following
            and interval '{{ label_horizon_minutes }} minutes' following
        ) as future_min_bikes_30,
        min(docks) over (
            partition by city, station_id
            order by snapshot_bucket_at_utc
            range between interval '{{ snapshot_step_minutes }} minutes' following
            and interval '{{ label_horizon_minutes }} minutes' following
        ) as future_min_docks_30,
        lead(bikes, {{ expected_future_snapshots }}) over (
            partition by city, station_id
            order by snapshot_bucket_at_utc
        ) as target_bikes_t30_raw,
        lead(docks, {{ expected_future_snapshots }}) over (
            partition by city, station_id
            order by snapshot_bucket_at_utc
        ) as target_docks_t30_raw
    from station_features sf
),
neighbor_aggregates as (
    select
        cur.city,
        cur.station_id,
        cur.snapshot_bucket_at_utc,
        sum(n.neighbor_weight * nbr.bikes::double precision) as nbr_bikes_weighted,
        sum(n.neighbor_weight * nbr.docks::double precision) as nbr_docks_weighted,
        coalesce(max(n.neighbor_count_within_radius), 0)::integer as neighbor_count_within_radius
    from station_features_windowed cur
    left join {{ ref('int_station_neighbors') }} n
        on cur.city = n.city
       and cur.station_id = n.station_id
    left join station_features_windowed nbr
        on nbr.city = n.city
       and nbr.station_id = n.neighbor_station_id
       and nbr.snapshot_bucket_at_utc = cur.snapshot_bucket_at_utc
    group by
        cur.city,
        cur.station_id,
        cur.snapshot_bucket_at_utc
),
assembled as (
    select
        sf.city,
        sf.dt,
        sf.station_id,
        sf.capacity,
        sf.lat,
        sf.lon,
        sf.bikes,
        sf.docks,
        sf.minutes_since_prev_snapshot,
        sf.util_bikes,
        sf.util_docks,
        sf.delta_bikes_5m,
        sf.delta_docks_5m,
        sf.roll15_net_bikes,
        sf.roll30_net_bikes,
        sf.roll60_net_bikes,
        sf.roll15_bikes_mean,
        sf.roll30_bikes_mean,
        sf.roll60_bikes_mean,
        na.nbr_bikes_weighted,
        na.nbr_docks_weighted,
        case
            when coalesce(na.neighbor_count_within_radius, 0) > 0 then 1
            else 0
        end as has_neighbors_within_radius,
        coalesce(na.neighbor_count_within_radius, 0)::integer as neighbor_count_within_radius,
        sf.hour,
        sf.dow,
        sf.is_weekend,
        sf.is_holiday,
        sf.temperature_c,
        sf.humidity_pct,
        sf.wind_speed_ms,
        sf.precipitation_mm,
        sf.weather_code,
        sf.hourly_temperature_c,
        sf.hourly_humidity_pct,
        sf.hourly_wind_speed_ms,
        sf.hourly_precipitation_mm,
        sf.hourly_precipitation_probability_pct,
        sf.hourly_weather_code,
        case
            when sf.future_snapshot_count = {{ expected_future_snapshots }}
                then sf.target_bikes_t30_raw
        end as target_bikes_t30,
        case
            when sf.future_snapshot_count = {{ expected_future_snapshots }}
                then sf.target_docks_t30_raw
        end as target_docks_t30,
        case
            when sf.future_snapshot_count = {{ expected_future_snapshots }}
                then case when sf.future_min_bikes_30 <= {{ stockout_threshold }} then 1 else 0 end
        end as y_stockout_bikes_30,
        case
            when sf.future_snapshot_count = {{ expected_future_snapshots }}
                then case when sf.future_min_docks_30 <= {{ stockout_threshold }} then 1 else 0 end
        end as y_stockout_docks_30,
        sf.snapshot_bucket_at_utc
    from station_features_windowed sf
    inner join neighbor_aggregates na
        on sf.city = na.city
       and sf.station_id = na.station_id
       and sf.snapshot_bucket_at_utc = na.snapshot_bucket_at_utc
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
    hourly_weather_code,
    target_bikes_t30,
    target_docks_t30,
    y_stockout_bikes_30,
    y_stockout_docks_30
from assembled
{% if is_incremental() %}
where snapshot_bucket_at_utc >= (select rebuild_from_utc from window_bounds)
  and snapshot_bucket_at_utc < (select rebuild_to_utc from window_bounds)
{% endif %}
