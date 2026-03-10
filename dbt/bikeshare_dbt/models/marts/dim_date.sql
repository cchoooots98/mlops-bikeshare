{{ config(
    materialized='table'
) }}

with holiday_dates as (
    select
        holiday_date,
        max(country_code) as country_code,
        bool_or(is_holiday) as is_holiday,
        max(holiday_name) as holiday_name
    from {{ ref('stg_holidays') }}
    group by holiday_date
),
status_bounds as (
    select
        min(snapshot_bucket_at_paris::date) as min_date,
        max(snapshot_bucket_at_paris::date) as max_date
    from {{ ref('stg_station_status') }}
),
holiday_bounds as (
    select
        min(holiday_date) as min_date,
        max(holiday_date) as max_date
    from holiday_dates
),
all_bounds as (
    select min(boundary_date) as min_date, max(boundary_date) as max_date
    from (
        select min_date as boundary_date from status_bounds
        union all
        select max_date as boundary_date from status_bounds
        union all
        select min_date as boundary_date from holiday_bounds
        union all
        select max_date as boundary_date from holiday_bounds
    ) bounds
    where boundary_date is not null
),
calendar as (
    select gs::date as date
    from all_bounds
    join lateral generate_series(all_bounds.min_date, all_bounds.max_date, interval '1 day') as gs
        on all_bounds.min_date is not null
),
final as (
    select
        to_char(c.date, 'YYYYMMDD')::int as date_id,
        c.date,
        extract(isodow from c.date)::smallint as day_of_week,
        extract(month from c.date)::smallint as month,
        extract(year from c.date)::smallint as year,
        (extract(isodow from c.date) in (6, 7)) as is_weekend,
        coalesce(h.is_holiday, false) as is_holiday,
        h.holiday_name
    from calendar c
    left join holiday_dates h
        on c.date = h.holiday_date
)
select *
from final
