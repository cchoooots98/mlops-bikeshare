{{ config(tags=['quality_gate']) }}

with source_rows as (
    select
        city,
        station_id,
        snapshot_bucket_at_utc,
        num_bikes_available as bikes,
        num_docks_available as docks
    from {{ ref('int_station_status_enriched') }}
    where snapshot_bucket_at_utc >= {{ runtime_window_start_utc_expr(default_lookback_hours=72) }}
      and snapshot_bucket_at_utc < {{ runtime_utc_expr('test_window_end_utc') }} + interval '30 minutes'
),
expected_targets as (
    select
        cur.city,
        cur.station_id,
        cur.snapshot_bucket_at_utc,
        fut.bikes as expected_target_bikes_t30,
        fut.docks as expected_target_docks_t30
    from source_rows cur
    left join source_rows fut
        on cur.city = fut.city
       and cur.station_id = fut.station_id
       and fut.snapshot_bucket_at_utc = cur.snapshot_bucket_at_utc + interval '30 minutes'
),
violations as (
    select
        f.city,
        f.station_id,
        f.dt,
        f.target_bikes_t30,
        e.expected_target_bikes_t30,
        f.target_docks_t30,
        e.expected_target_docks_t30
    from {{ ref('feat_station_snapshot_5min') }} f
    inner join expected_targets e
        on f.city = e.city
       and f.station_id = e.station_id
       and {{ feature_dt_to_utc('f.dt') }} = e.snapshot_bucket_at_utc
    where f.target_bikes_t30 is not null
       or f.target_docks_t30 is not null
)
select *
from violations
where coalesce(target_bikes_t30, -1) <> coalesce(expected_target_bikes_t30, -1)
   or coalesce(target_docks_t30, -1) <> coalesce(expected_target_docks_t30, -1)
