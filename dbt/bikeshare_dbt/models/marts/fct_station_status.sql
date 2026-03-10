{{ config(
    materialized='table'
) }}

with status_ranked as (
    select
        city,
        station_id,
        station_key,
        snapshot_bucket_at_utc,
        snapshot_bucket_at_paris,
        last_reported_at_utc,
        last_reported_at_paris,
        num_bikes_available,
        num_docks_available,
        is_renting,
        is_returning,
        ingested_at_utc,
        source_last_updated,
        run_id,
        row_number() over (
            partition by city, station_id, snapshot_bucket_at_utc
            order by ingested_at_utc desc, source_last_updated desc, run_id desc
        ) as row_num
    from {{ ref('stg_station_status') }}
),
status_deduped as (
    select
        city,
        station_id,
        station_key,
        snapshot_bucket_at_utc,
        snapshot_bucket_at_paris,
        last_reported_at_utc,
        last_reported_at_paris,
        num_bikes_available,
        num_docks_available,
        is_renting,
        is_returning
    from status_ranked
    where row_num = 1
),
station_joined as (
    select
        concat(
            s.city,
            '|',
            to_char(s.snapshot_bucket_at_utc, 'YYYY-MM-DD HH24:MI:SSOF'),
            '|',
            s.station_id
        ) as fact_station_status_key,
        s.city,
        s.station_id,
        s.station_key,
        d.station_version_key,
        s.snapshot_bucket_at_utc,
        s.snapshot_bucket_at_paris,
        s.last_reported_at_utc,
        s.last_reported_at_paris,
        s.num_bikes_available,
        s.num_docks_available,
        s.is_renting,
        s.is_returning
    from status_deduped s
    left join {{ ref('dim_station') }} d
        on s.station_key = d.station_key
       and s.snapshot_bucket_at_utc >= d.valid_from_utc
       and (s.snapshot_bucket_at_utc < d.valid_to_utc or d.valid_to_utc is null)
),
date_time_joined as (
    select
        s.fact_station_status_key,
        s.city,
        s.station_id,
        s.station_key,
        s.station_version_key,
        s.snapshot_bucket_at_utc,
        s.snapshot_bucket_at_paris,
        dd.date_id,
        dt.time_id,
        s.last_reported_at_utc,
        s.last_reported_at_paris,
        s.num_bikes_available,
        s.num_docks_available,
        s.is_renting,
        s.is_returning
    from station_joined s
    left join {{ ref('dim_date') }} dd
        on s.snapshot_bucket_at_paris::date = dd.date
    left join {{ ref('dim_time') }} dt
        on (
            extract(hour from s.snapshot_bucket_at_paris)::integer * 60
            + extract(minute from s.snapshot_bucket_at_paris)::integer
        ) = dt.time_id
)
select
    fact_station_status_key,
    city,
    station_id,
    station_key,
    station_version_key,
    snapshot_bucket_at_utc,
    snapshot_bucket_at_paris,
    date_id,
    time_id,
    last_reported_at_utc,
    last_reported_at_paris,
    num_bikes_available,
    num_docks_available,
    is_renting,
    is_returning
from date_time_joined
