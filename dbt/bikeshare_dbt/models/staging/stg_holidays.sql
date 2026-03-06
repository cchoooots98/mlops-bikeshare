with src as (
    select * from {{ source('raw_staging', 'stg_holidays') }}
)
select
    run_id::text as run_id,
    ingested_at::timestamptz as ingested_at_utc,
    (ingested_at::timestamptz at time zone 'Europe/Paris')::timestamp as ingested_at_paris,
    country_code::text as country_code,
    holiday_date::date as holiday_date,
    is_holiday::boolean as is_holiday,
    holiday_name::text as holiday_name,
    concat(country_code::text, '|', holiday_date::text) as holiday_pk
from src

