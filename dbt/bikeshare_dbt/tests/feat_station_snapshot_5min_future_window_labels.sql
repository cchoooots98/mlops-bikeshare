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
mature_feature_rows as (
    select
        city,
        station_id,
        dt,
        {{ feature_dt_to_utc('dt') }} as snapshot_bucket_at_utc,
        y_stockout_bikes_30,
        y_stockout_docks_30
    from {{ ref('feat_station_snapshot_5min') }}
    where {{ feature_dt_to_utc('dt') }} >= {{ runtime_window_start_utc_expr(default_lookback_hours=72) }}
      and {{ feature_dt_to_utc('dt') }} < {{ runtime_utc_expr('test_window_end_utc') }}
      and y_stockout_bikes_30 is not null
      and y_stockout_docks_30 is not null
),
expected as (
    select
        cur.city,
        cur.station_id,
        cur.snapshot_bucket_at_utc,
        case
            when min(fut.bikes) <= {{ var('stockout_threshold', 2) | int }} then 1
            else 0
        end as expected_bike_label,
        case
            when min(fut.docks) <= {{ var('stockout_threshold', 2) | int }} then 1
            else 0
        end as expected_dock_label
    from mature_feature_rows cur
    left join source_rows fut
        on cur.city = fut.city
       and cur.station_id = fut.station_id
       and fut.snapshot_bucket_at_utc > cur.snapshot_bucket_at_utc
       and fut.snapshot_bucket_at_utc <= cur.snapshot_bucket_at_utc + interval '30 minutes'
    group by cur.city, cur.station_id, cur.snapshot_bucket_at_utc
)
select
    cur.city,
    cur.station_id,
    cur.dt,
    cur.y_stockout_bikes_30,
    expected.expected_bike_label,
    cur.y_stockout_docks_30,
    expected.expected_dock_label
from mature_feature_rows cur
inner join expected
    on cur.city = expected.city
   and cur.station_id = expected.station_id
   and cur.snapshot_bucket_at_utc = expected.snapshot_bucket_at_utc
where cur.y_stockout_bikes_30 <> expected.expected_bike_label
   or cur.y_stockout_docks_30 <> expected.expected_dock_label
