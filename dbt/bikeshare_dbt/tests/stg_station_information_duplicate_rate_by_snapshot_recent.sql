{% set lookback_days = env_var('DBT_STATION_INFORMATION_QUALITY_LOOKBACK_DAYS', '2') | int %}
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
        snapshot_bucket_at,
        station_id,
        count(*) as row_count
    from {{ source('raw_staging', 'stg_station_information') }}
    where snapshot_bucket_at >= current_timestamp - interval '{{ lookback_days }} days'
    group by city, snapshot_bucket_at, station_id
),
rates as (
    select
        city,
        snapshot_bucket_at as bucket_at_utc,
        coalesce(
            sum(greatest(row_count - 1, 0))::double precision
            / nullif(sum(row_count)::double precision, 0.0),
            0.0
        ) as observed_rate
    from grouped_rows
    group by city, snapshot_bucket_at
)
select *
from rates
where observed_rate > 0.0
