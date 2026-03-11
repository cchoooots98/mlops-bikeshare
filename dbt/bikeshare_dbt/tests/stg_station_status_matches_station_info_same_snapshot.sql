with unmatched_status as (
    select
        s.city,
        s.station_id,
        s.snapshot_bucket_at_utc
    from {{ ref('stg_station_status') }} s
    left join {{ ref('stg_station_information') }} i
        on s.city = i.city
       and s.station_id = i.station_id
       and i.snapshot_bucket_at_utc <= s.snapshot_bucket_at_utc
    where i.station_info_pk is null
)
select *
from unmatched_status
