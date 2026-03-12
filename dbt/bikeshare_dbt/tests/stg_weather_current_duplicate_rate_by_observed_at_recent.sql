{% set lookback_hours = env_var('DBT_QUALITY_LOOKBACK_HOURS', '2') | int %}
{% set warn_rate = env_var('DBT_DUPLICATE_WARN_RATE', '0.001') %}
{% set error_rate = env_var('DBT_DUPLICATE_ERROR_RATE', '0.01') %}

{{ config(
    tags=['quality_gate'],
    fail_calc='coalesce(max(observed_rate), 0.0)',
    warn_if='> ' ~ warn_rate,
    error_if='> ' ~ error_rate
) }}

with grouped_rows as (
    select
        city,
        observed_at,
        count(*) as row_count
    from {{ source('raw_staging', 'stg_weather_current') }}
    where observed_at >= current_timestamp - interval '{{ lookback_hours }} hours'
    group by city, observed_at
),
rates as (
    select
        city,
        observed_at,
        coalesce(
            sum(greatest(row_count - 1, 0))::double precision
            / nullif(sum(row_count)::double precision, 0.0),
            0.0
        ) as observed_rate
    from grouped_rows
    group by city, observed_at
)
select *
from rates
where observed_rate > 0.0
