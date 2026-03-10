{{ config(
    materialized='table'
) }}

with ranked_station_info as (
    select
        station_key,
        city,
        station_id,
        station_name,
        latitude,
        longitude,
        capacity,
        snapshot_bucket_at_utc,
        row_number() over (
            partition by station_key, snapshot_bucket_at_utc
            order by ingested_at_utc desc, source_last_updated desc, run_id desc
        ) as row_num
    from {{ ref('stg_station_information') }}
),
station_info_deduped as (
    select
        station_key,
        city,
        station_id,
        station_name,
        latitude,
        longitude,
        capacity,
        snapshot_bucket_at_utc
    from ranked_station_info
    where row_num = 1
),
station_change_candidates as (
    select
        station_key,
        city,
        station_id,
        station_name,
        latitude,
        longitude,
        capacity,
        snapshot_bucket_at_utc,
        md5(
            concat_ws(
                '||',
                coalesce(station_name, ''),
                coalesce(latitude::text, ''),
                coalesce(longitude::text, ''),
                coalesce(capacity::text, '')
            )
        ) as attribute_hash
    from station_info_deduped
),
station_change_points as (
    select
        station_key,
        city,
        station_id,
        station_name,
        latitude,
        longitude,
        capacity,
        snapshot_bucket_at_utc
    from (
        select
            *,
            lag(attribute_hash) over (
                partition by station_key
                order by snapshot_bucket_at_utc
            ) as previous_attribute_hash
        from station_change_candidates
    ) changes
    where previous_attribute_hash is null
       or previous_attribute_hash <> attribute_hash
),
station_scd2 as (
    select
        concat(
            station_key,
            '|',
            to_char(snapshot_bucket_at_utc, 'YYYY-MM-DD HH24:MI:SSOF')
        ) as station_version_key,
        station_key,
        city,
        station_id,
        station_name,
        latitude,
        longitude,
        capacity,
        snapshot_bucket_at_utc as valid_from_utc,
        lead(snapshot_bucket_at_utc) over (
            partition by station_key
            order by snapshot_bucket_at_utc
        ) as valid_to_utc
    from station_change_points
)
select
    station_version_key,
    station_key,
    city,
    station_id,
    station_name,
    latitude,
    longitude,
    capacity,
    valid_from_utc,
    valid_to_utc,
    (valid_to_utc is null) as is_current
from station_scd2
