with duplicates as (
    select
        city,
        station_id,
        dt,
        count(*) as row_count
    from {{ ref('feat_station_snapshot_5min') }}
    group by city, station_id, dt
    having count(*) > 1
)
select *
from duplicates
