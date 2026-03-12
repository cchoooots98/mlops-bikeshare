{{ config(severity='warn', tags=['quality_gate']) }}

{% set max_missing_weather_ratio = var('int_station_status_enriched_max_missing_weather_context_ratio', 0.05) | float %}

with coverage_by_city as (
    select
        city,
        count(*) as total_rows,
        sum(case when has_weather_context = 0 then 1 else 0 end) as missing_weather_rows
    from {{ ref('int_station_status_enriched') }}
    group by city
),
violations as (
    select
        city,
        total_rows,
        missing_weather_rows,
        missing_weather_rows::double precision / nullif(total_rows, 0)::double precision as missing_weather_ratio,
        {{ max_missing_weather_ratio }}::double precision as max_missing_weather_ratio
    from coverage_by_city
)
select *
from violations
where total_rows > 0
  and missing_weather_ratio > max_missing_weather_ratio
