{{ config(severity='warn', tags=['diagnostic']) }}

select
    city,
    station_id,
    snapshot_bucket_at_utc,
    num_bikes_available,
    num_docks_available,
    capacity
from {{ ref('int_station_status_enriched') }}
where num_bikes_available + num_docks_available > capacity
