{{ config(tags=['deep_quality']) }}

with comparison_window as (
    select
        {{ runtime_window_start_utc_expr(default_lookback_hours=72) }} as window_start_utc,
        {{ runtime_utc_expr('test_window_end_utc') }} as window_end_utc,
        {{ runtime_utc_expr('test_window_end_utc') }} + interval '30 minutes' as future_window_end_utc
),
feature_rows as (
    select
        f.city,
        f.station_id,
        f.dt,
        {{ feature_dt_to_utc('f.dt') }} as snapshot_bucket_at_utc,
        f.bikes,
        f.docks,
        f.target_bikes_t30,
        f.target_docks_t30
    from {{ ref('feat_station_snapshot_5min') }} f
    cross join comparison_window cw
    where {{ feature_dt_to_utc('f.dt') }} >= cw.window_start_utc
      and {{ feature_dt_to_utc('f.dt') }} < cw.future_window_end_utc
),
current_rows as (
    select *
    from feature_rows
    where snapshot_bucket_at_utc < (select window_end_utc from comparison_window)
      and (
            target_bikes_t30 is not null
         or target_docks_t30 is not null
      )
),
expected_targets as (
    select
        cur.city,
        cur.station_id,
        cur.dt,
        max(fut.snapshot_bucket_at_utc) as expected_target_snapshot_bucket_at_utc
    from current_rows cur
    left join feature_rows fut
        on cur.city = fut.city
       and cur.station_id = fut.station_id
       and fut.snapshot_bucket_at_utc > cur.snapshot_bucket_at_utc
       and fut.snapshot_bucket_at_utc <= cur.snapshot_bucket_at_utc + interval '30 minutes'
    group by cur.city, cur.station_id, cur.dt
),
expected_target_values as (
    select
        expected.city,
        expected.station_id,
        expected.dt,
        fut.bikes as expected_target_bikes_t30,
        fut.docks as expected_target_docks_t30
    from expected_targets expected
    left join feature_rows fut
        on expected.city = fut.city
       and expected.station_id = fut.station_id
       and expected.expected_target_snapshot_bucket_at_utc = fut.snapshot_bucket_at_utc
)
select
    cur.city,
    cur.station_id,
    cur.dt,
    cur.target_bikes_t30,
    expected.expected_target_bikes_t30,
    cur.target_docks_t30,
    expected.expected_target_docks_t30
from current_rows cur
inner join expected_target_values expected
    on cur.city = expected.city
   and cur.station_id = expected.station_id
   and cur.dt = expected.dt
where coalesce(cur.target_bikes_t30, -1) <> coalesce(expected.expected_target_bikes_t30, -1)
   or coalesce(cur.target_docks_t30, -1) <> coalesce(expected.expected_target_docks_t30, -1)
