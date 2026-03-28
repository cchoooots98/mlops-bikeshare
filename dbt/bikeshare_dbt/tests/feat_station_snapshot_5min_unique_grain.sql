{{ config(tags=['hf_smoke']) }}

with duplicates as (
    select
        city,
        station_id,
        dt,
        count(*) as row_count
    from {{ ref('feat_station_snapshot_5min') }}
    where {{ feature_dt_to_utc('dt') }} >= {{ runtime_window_start_utc_expr(default_lookback_hours=24) }}
      and {{ feature_dt_to_utc('dt') }} < {{ runtime_utc_expr('test_window_end_utc') }}
    group by city, station_id, dt
    having count(*) > 1
)
select *
from duplicates
