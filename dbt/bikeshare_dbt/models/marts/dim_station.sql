{{ config(
    materialized='table'
) }}

with station_info_source as (
    select
        station_key,
        city,
        station_id,
        station_name,
        latitude,
        longitude,
        capacity,
        snapshot_bucket_at_utc
    from {{ ref('stg_station_information') }}
    where latitude between -90 and 90
      and longitude between -180 and 180
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
    from station_info_source
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
        {{ station_snapshot_key('city', 'station_id', 'snapshot_bucket_at_utc') }} as station_version_key,
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
