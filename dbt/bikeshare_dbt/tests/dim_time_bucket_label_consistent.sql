{{ config(tags=['deep_quality']) }}

select
    time_id,
    hour,
    minute,
    minute_of_day,
    bucket_label
from {{ ref('dim_time') }}
where minute_of_day <> time_id
   or minute_of_day <> hour * 60 + minute
   or minute not in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55)
   or bucket_label <> lpad(hour::text, 2, '0') || ':' || lpad(minute::text, 2, '0')
