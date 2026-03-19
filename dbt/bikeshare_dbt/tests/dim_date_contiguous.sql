{{ config(tags=['deep_quality']) }}

with ordered_dates as (
    select
        date,
        lead(date) over (order by date) as next_date
    from {{ ref('dim_date') }}
),
gaps as (
    select
        date,
        next_date
    from ordered_dates
    where next_date is not null
      and next_date <> date + 1
)
select *
from gaps
