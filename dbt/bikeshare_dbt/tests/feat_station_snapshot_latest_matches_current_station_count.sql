{{ config(tags=['hf_smoke']) }}

with current_station_counts as (
    select
        city,
        count(*) as current_station_count
    from {{ ref('dim_station') }}
    where is_current = true
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
    c.city,
    c.current_station_count,
    coalesce(l.latest_station_count, 0) as latest_station_count
from current_station_counts c
left join latest_counts l
    on c.city = l.city
where coalesce(l.latest_station_count, 0) <> c.current_station_count
