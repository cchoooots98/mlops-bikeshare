with duplicates as (
    select
        city,
        station_id,
        count(*) as row_count
    from {{ ref('feat_station_snapshot_latest') }}
    group by city, station_id
    having count(*) > 1
)
select *
from duplicates
