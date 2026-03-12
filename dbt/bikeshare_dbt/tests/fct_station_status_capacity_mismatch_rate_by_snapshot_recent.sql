{% set lookback_hours = env_var('DBT_QUALITY_LOOKBACK_HOURS', '2') | int %}
{% set warn_rate = env_var('DBT_CAPACITY_MISMATCH_WARN_RATE', '0.005') %}
{% set error_rate = env_var('DBT_CAPACITY_MISMATCH_ERROR_RATE', '0.02') %}

{{ config(
    tags=['quality_gate'],
    fail_calc='coalesce(max(observed_rate), 0.0)',
    warn_if='> ' ~ warn_rate,
    error_if='> ' ~ error_rate
) }}

with recent_rows as (
    select
        f.city,
        f.snapshot_bucket_at_utc,
        case
            when f.num_bikes_available + f.num_docks_available > d.capacity then 1
            else 0
        end as mismatch_flag
    from {{ ref('fct_station_status') }} f
    inner join {{ ref('dim_station') }} d
        on f.station_version_key = d.station_version_key
    where f.snapshot_bucket_at_utc >= current_timestamp - interval '{{ lookback_hours }} hours'
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
