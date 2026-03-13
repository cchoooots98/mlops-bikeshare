{{ config(tags=['quality_gate']) }}

select
    weather_key
from {{ ref('dim_weather') }}
where hourly_forecast_at is not null
  and hourly_forecast_at <= observed_at
