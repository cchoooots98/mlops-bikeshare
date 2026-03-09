with src as (
    select * from {{ source('raw_staging', 'stg_station_status') }}
)
select
    run_id::text as run_id,
    ingested_at::timestamptz as ingested_at_utc,
    (ingested_at::timestamptz at time zone 'Europe/Paris')::timestamp as ingested_at_paris,
    source_last_updated::bigint as source_last_updated,
    city::text as city,
    station_id::text as station_id,
    last_reported_at::timestamptz as last_reported_at_utc,
    (last_reported_at::timestamptz at time zone 'Europe/Paris')::timestamp as last_reported_at_paris,
    num_bikes_available::integer as num_bikes_available,
    num_docks_available::integer as num_docks_available,
    is_renting::smallint as is_renting,
    is_returning::smallint as is_returning,
    concat(city::text, '|', station_id::text) as station_key,
    concat(city::text, '|', run_id::text, '|', station_id::text) as station_status_pk
from src
