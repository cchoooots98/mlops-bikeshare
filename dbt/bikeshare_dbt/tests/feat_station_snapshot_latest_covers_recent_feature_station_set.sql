{{ config(tags=['quality_gate']) }}

with recent_feature_station_set as (
    select distinct
        city,
        station_id
    from {{ ref('feat_station_snapshot_5min') }}
    where {{ feature_dt_to_utc('dt') }} >= {{ runtime_window_start_utc_expr(default_lookback_hours=72) }}
      and {{ feature_dt_to_utc('dt') }} < {{ runtime_utc_expr('test_window_end_utc') }}
),
latest_station_set as (
    select
        city,
        station_id
    from {{ ref('feat_station_snapshot_latest') }}
),
missing_from_latest as (
    select
        r.city,
        r.station_id
    from recent_feature_station_set r
    left join latest_station_set l
        on r.city = l.city
       and r.station_id = l.station_id
    where l.station_id is null
)
select *
from missing_from_latest
