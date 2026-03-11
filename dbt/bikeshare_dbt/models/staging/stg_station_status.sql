{{ config(
    materialized='view'
) }}

with src as (
    select * from {{ source('raw_staging', 'stg_station_status') }}
),
ranked as (
    select
        run_id::text as run_id,
        ingested_at::timestamptz as ingested_at_utc,
        (ingested_at::timestamptz at time zone 'Europe/Paris')::timestamp as ingested_at_paris,
        source_last_updated::bigint as source_last_updated,
        city::text as city,
        snapshot_bucket_at::timestamptz as snapshot_bucket_at_utc,
        (snapshot_bucket_at::timestamptz at time zone 'Europe/Paris')::timestamp as snapshot_bucket_at_paris,
        station_id::text as station_id,
        last_reported_at::timestamptz as last_reported_at_utc,
        (last_reported_at::timestamptz at time zone 'Europe/Paris')::timestamp as last_reported_at_paris,
        num_bikes_available::integer as num_bikes_available,
        num_docks_available::integer as num_docks_available,
        is_renting::smallint as is_renting,
        is_returning::smallint as is_returning,
        row_number() over (
            partition by
                city::text,
                snapshot_bucket_at::timestamptz,
                station_id::text
            order by
                source_last_updated::bigint desc,
                ingested_at::timestamptz desc,
                run_id::text desc
        ) as row_num
    from src
)
select
    run_id,
    ingested_at_utc,
    ingested_at_paris,
    source_last_updated,
    city,
    snapshot_bucket_at_utc,
    snapshot_bucket_at_paris,
    station_id,
    last_reported_at_utc,
    last_reported_at_paris,
    num_bikes_available,
    num_docks_available,
    is_renting,
    is_returning,
    concat(city, '|', station_id) as station_key,
    {{ station_snapshot_key('city', 'station_id', 'snapshot_bucket_at_utc') }} as station_status_pk
from ranked
where row_num = 1
