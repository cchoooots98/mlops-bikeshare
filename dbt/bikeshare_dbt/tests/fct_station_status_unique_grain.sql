with duplicates as (
    select
        city,
        station_id,
        snapshot_bucket_at_utc,
        count(*) as row_count
    from {{ ref('fct_station_status') }}
    group by city, station_id, snapshot_bucket_at_utc
    having count(*) > 1
)
select *
from duplicates
