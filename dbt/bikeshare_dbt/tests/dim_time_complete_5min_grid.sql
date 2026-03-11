with stats as (
    select
        count(*) as row_count,
        min(time_id) as min_time_id,
        max(time_id) as max_time_id
    from {{ ref('dim_time') }}
),
violations as (
    select
        row_count,
        min_time_id,
        max_time_id
    from stats
    where row_count <> 288
       or min_time_id <> 0
       or max_time_id <> 1435
)
select *
from violations
