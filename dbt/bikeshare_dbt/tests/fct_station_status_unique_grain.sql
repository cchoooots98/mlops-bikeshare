{{ config(tags=['hf_hotpath_smoke']) }}

with duplicates as (
    select
        city,
        station_id,
        snapshot_bucket_at_utc,
        count(*) as row_count
    from {{ ref('fct_station_status') }}
    where snapshot_bucket_at_utc >= {{ runtime_window_start_utc_expr(default_lookback_hours=2) }}
      and snapshot_bucket_at_utc < {{ runtime_utc_expr('test_window_end_utc') }}
    group by city, station_id, snapshot_bucket_at_utc
    having count(*) > 1
)
select *
from duplicates
