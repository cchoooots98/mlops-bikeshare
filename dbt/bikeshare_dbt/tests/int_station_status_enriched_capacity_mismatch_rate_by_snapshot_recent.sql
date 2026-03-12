{% set lookback_hours = env_var('DBT_QUALITY_LOOKBACK_HOURS', '2') | int %}
{% set warn_rate = env_var('DBT_CAPACITY_MISMATCH_WARN_RATE', '0.005') | float %}
{% set error_rate = env_var('DBT_CAPACITY_MISMATCH_ERROR_RATE', '0.02') | float %}
{% set rate_scale = 1000000 %}
{% set warn_rate_scaled = (warn_rate * rate_scale) | round(0) | int %}
{% set error_rate_scaled = (error_rate * rate_scale) | round(0) | int %}

{{ config(
    tags=['quality_gate'],
    fail_calc='coalesce(max((observed_rate * ' ~ rate_scale ~ ')::bigint), 0)',
    warn_if='> ' ~ warn_rate_scaled,
    error_if='> ' ~ error_rate_scaled
) }}

with recent_rows as (
    select
        city,
        snapshot_bucket_at_utc,
        case
            when num_bikes_available + num_docks_available > capacity then 1
            else 0
        end as mismatch_flag
    from {{ ref('int_station_status_enriched') }}
    where snapshot_bucket_at_utc >= current_timestamp - interval '{{ lookback_hours }} hours'
),
rates as (
    select
        city,
        snapshot_bucket_at_utc,
        avg(mismatch_flag::double precision) as observed_rate
    from recent_rows
    group by city, snapshot_bucket_at_utc
)
select *
from rates
where observed_rate > 0.0
