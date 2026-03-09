{{ config(
    materialized='table'
) }}

with ranked_station_info as (
    select
        city,
        station_id,
        station_name,
        latitude,
        longitude,
        capacity,
        ingested_at_utc,
        source_last_updated,
        run_id,
        row_number() over (
            partition by city, station_id
            order by ingested_at_utc desc, source_last_updated desc, run_id desc
        ) as row_num
    from {{ ref('stg_station_information') }}
),
latest_station_info as (
    select
        city,
        station_id,
        station_name,
        latitude,
        longitude,
        capacity
    from ranked_station_info
    where row_num = 1
)
select
    concat(city, '|', station_id) as station_key,
    city,
    station_id,
    station_name,
    latitude,
    longitude,
    capacity
from latest_station_info
