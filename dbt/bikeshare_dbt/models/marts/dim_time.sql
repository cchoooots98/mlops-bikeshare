{{ config(
    materialized='view'
) }}

with bucket_index as (
    select
        generate_series(0, 287) as bucket_num
),

time_base as (
    select
        bucket_num,
        bucket_num * 5 as minute_of_day
    from bucket_index
),

final as (
    select
        minute_of_day as time_id,
        (minute_of_day / 60)::integer as hour,
        (minute_of_day % 60)::integer as minute,
        minute_of_day,
        lpad(((minute_of_day / 60)::integer)::text, 2, '0')
        || ':'
        || lpad(((minute_of_day % 60)::integer)::text, 2, '0') as bucket_label
    from time_base
)

select
    time_id,
    hour,
    minute,
    minute_of_day,
    bucket_label
from final
order by time_id
