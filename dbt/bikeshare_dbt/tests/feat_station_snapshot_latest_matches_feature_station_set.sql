{{ config(tags=['hf_smoke']) }}

with history_station_set as (
    select
        city,
        station_id
    from {{ ref('feat_station_snapshot_5min') }}
    group by city, station_id
),
latest_station_set as (
    select
        city,
        station_id
    from {{ ref('feat_station_snapshot_latest') }}
),
only_in_history as (
    select
        h.city,
        h.station_id,
        'missing_from_latest' as mismatch_type
    from history_station_set h
    left join latest_station_set l
        on h.city = l.city
       and h.station_id = l.station_id
    where l.station_id is null
),
only_in_latest as (
    select
        l.city,
        l.station_id,
        'unexpected_in_latest' as mismatch_type
    from latest_station_set l
    left join history_station_set h
        on h.city = l.city
       and h.station_id = l.station_id
    where h.station_id is null
)
select
    city,
    station_id,
    mismatch_type
from only_in_history
union all
select
    city,
    station_id,
    mismatch_type
from only_in_latest
