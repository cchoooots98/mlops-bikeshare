{{ config(
    materialized='view'
) }}

with src as (
    select * from {{ source('raw_staging', 'stg_station_information') }}
)
select
    run_id::text as run_id,
    ingested_at::timestamptz as ingested_at_utc,
    (ingested_at::timestamptz at time zone 'Europe/Paris')::timestamp as ingested_at_paris,
    source_last_updated::bigint as source_last_updated,
    city::text as city,
    station_id::text as station_id,
    name::text as station_name,
    lat::double precision as latitude,
    lon::double precision as longitude,
    capacity::integer as capacity,
    concat(city::text, '|', station_id::text) as station_key,
    concat(city::text, '|', run_id::text, '|', station_id::text) as station_info_pk
from src
