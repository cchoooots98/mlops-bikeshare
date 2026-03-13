{{ config(tags=['quality_gate']) }}

with unmatched_status as (
    select
        s.city,
        s.station_id,
        s.snapshot_bucket_at_utc
    from {{ ref('stg_station_status') }} s
    where not exists (
        select 1
        from {{ ref('stg_station_information') }} i
        where s.city = i.city
          and s.station_id = i.station_id
          and i.snapshot_bucket_at_utc <= s.snapshot_bucket_at_utc
    )
)
select *
from unmatched_status
