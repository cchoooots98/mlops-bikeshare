{{ config(tags=['quality_gate']) }}

with latest_dim_weather as (
    select
        max(snapshot_bucket_at_utc) as max_snapshot_bucket_at_utc
    from {{ ref('dim_weather') }}
)

select
    c.weather_current_pk
from {{ ref('stg_weather_current') }} c
cross join latest_dim_weather ldw
left join {{ ref('dim_weather') }} d
    on d.weather_key = c.weather_current_pk
where ldw.max_snapshot_bucket_at_utc is not null
  and c.snapshot_bucket_at_utc <= ldw.max_snapshot_bucket_at_utc
  and d.weather_key is null
