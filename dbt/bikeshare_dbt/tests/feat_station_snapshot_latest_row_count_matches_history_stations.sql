{{ config(tags=['quality_gate']) }}

with history_counts as (
    select
        city,
        count(distinct station_id) as history_station_count
    from {{ ref('feat_station_snapshot_5min') }}
    group by city
),
latest_counts as (
    select
        city,
        count(*) as latest_station_count
    from {{ ref('feat_station_snapshot_latest') }}
    group by city
)
select
    h.city,
    h.history_station_count,
    coalesce(l.latest_station_count, 0) as latest_station_count
from history_counts h
left join latest_counts l
    on h.city = l.city
where coalesce(l.latest_station_count, 0) <> h.history_station_count
