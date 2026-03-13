{{ config(tags=['quality_gate']) }}

select
    weather_key,
    observed_at,
    hourly_forecast_at
from {{ ref('dim_weather') }}
where hourly_forecast_at is not null
  and hourly_forecast_at > observed_at + interval '60 minutes'
