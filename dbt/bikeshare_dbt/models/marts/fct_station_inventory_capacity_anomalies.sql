{{ config(
    materialized='view'
) }}

select
    f.city,
    f.station_id,
    f.station_key,
    f.station_version_key,
    f.snapshot_bucket_at_utc,
    f.snapshot_bucket_at_paris,
    f.num_bikes_available,
    f.num_docks_available,
    d.capacity,
    (f.num_bikes_available + f.num_docks_available - d.capacity) as over_capacity_by
from {{ ref('fct_station_status') }} f
inner join {{ ref('dim_station') }} d
    on f.station_version_key = d.station_version_key
where f.num_bikes_available + f.num_docks_available > d.capacity
