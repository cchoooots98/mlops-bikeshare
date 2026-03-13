{% set station_status_rebuild_lookback_minutes = var('station_status_rebuild_lookback_minutes', 15) | int %}
{% set existing_relation = adapter.get_relation(database=this.database, schema=this.schema, identifier=this.identifier) %}
{% set run_recent_window_overwrite = existing_relation is not none and not flags.FULL_REFRESH %}
{% set recent_window_overwrite_sql %}
{% if run_recent_window_overwrite %}
delete from {{ this }}
where snapshot_bucket_at_utc >= {{ runtime_utc_expr('hotpath_window_end_utc') }}
    - interval '{{ station_status_rebuild_lookback_minutes }} minutes'
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
    pre_hook=[recent_window_overwrite_sql]
) }}

with window_bounds as (
    select
        {{ runtime_utc_expr('hotpath_window_end_utc') }} as rebuild_to_utc,
        {{ runtime_utc_expr('hotpath_window_end_utc') }}
            - interval '{{ station_status_rebuild_lookback_minutes }} minutes' as rebuild_from_utc
),
status_source as (
    select
        city,
        station_id,
        station_key,
        snapshot_bucket_at_utc,
        snapshot_bucket_at_paris,
        last_reported_at_utc,
        last_reported_at_paris,
        num_bikes_available,
        num_docks_available,
        is_renting,
        is_returning
    from {{ ref('stg_station_status') }}
    {% if is_incremental() %}
    cross join window_bounds wb
    where snapshot_bucket_at_utc >= wb.rebuild_from_utc
      and snapshot_bucket_at_utc < wb.rebuild_to_utc
    {% endif %}
),
station_joined as (
    select
        {{ station_snapshot_key('s.city', 's.station_id', 's.snapshot_bucket_at_utc') }} as fact_station_status_key,
        s.city,
        s.station_id,
        s.station_key,
        d.station_version_key,
        s.snapshot_bucket_at_utc,
        s.snapshot_bucket_at_paris,
        s.last_reported_at_utc,
        s.last_reported_at_paris,
        s.num_bikes_available,
        s.num_docks_available,
        s.is_renting,
        s.is_returning
    from status_source s
    left join {{ ref('dim_station') }} d
        on s.station_key = d.station_key
       and s.snapshot_bucket_at_utc >= d.valid_from_utc
       and (s.snapshot_bucket_at_utc < d.valid_to_utc or d.valid_to_utc is null)
),
validated_station_inventory as (
    select
        fact_station_status_key,
        city,
        station_id,
        station_key,
        station_version_key,
        snapshot_bucket_at_utc,
        snapshot_bucket_at_paris,
        last_reported_at_utc,
        last_reported_at_paris,
        num_bikes_available,
        num_docks_available,
        is_renting,
        is_returning
    from station_joined
    where station_version_key is not null
),
date_time_joined as (
    select
        s.fact_station_status_key,
        s.city,
        s.station_id,
        s.station_key,
        s.station_version_key,
        s.snapshot_bucket_at_utc,
        s.snapshot_bucket_at_paris,
        dd.date_id,
        dt.time_id,
        s.last_reported_at_utc,
        s.last_reported_at_paris,
        s.num_bikes_available,
        s.num_docks_available,
        s.is_renting,
        s.is_returning
    from validated_station_inventory s
    left join {{ ref('dim_date') }} dd
        on s.snapshot_bucket_at_paris::date = dd.date
    left join {{ ref('dim_time') }} dt
        on (
            extract(hour from s.snapshot_bucket_at_paris)::integer * 60
            + extract(minute from s.snapshot_bucket_at_paris)::integer
        ) = dt.time_id
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
    is_returning
from date_time_joined
