{{ config(severity='warn') }}

select
    f.city,
    f.station_id,
    f.snapshot_bucket_at_utc,
    f.num_bikes_available,
    f.num_docks_available,
    d.capacity
from {{ ref('fct_station_status') }} f
inner join {{ ref('dim_station') }} d
    on f.station_version_key = d.station_version_key
where f.num_bikes_available + f.num_docks_available > d.capacity
