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
),
latest_rows as (
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
        concat(city, '|', station_id) as station_key
    from ranked
    where row_num = 1
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
    station_key,
    {{ station_snapshot_key('city', 'station_id', 'snapshot_bucket_at_utc') }} as station_status_pk
from latest_rows
where nullif(trim(station_id), '') is not null
  and last_reported_at_utc is not null
  and num_bikes_available is not null
  and num_bikes_available >= 0
  and num_docks_available is not null
  and num_docks_available >= 0
  and is_renting in (0, 1)
  and is_returning in (0, 1)
