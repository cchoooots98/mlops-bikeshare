{{ config(tags=['hf_hotpath_smoke']) }}

with duplicates as (
    select
        city,
        station_id,
        snapshot_bucket_at_utc,
        count(*) as row_count
    from {{ ref('int_station_status_enriched') }}
    group by city, station_id, snapshot_bucket_at_utc
    having count(*) > 1
)
select *
from duplicates
