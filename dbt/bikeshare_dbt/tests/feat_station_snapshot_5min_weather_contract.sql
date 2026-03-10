with columns_present as (
    select column_name
    from information_schema.columns
    where table_schema = '{{ target.schema }}'
      and table_name = 'feat_station_snapshot_5min'
),
required_columns as (
    select 'temperature_c' as column_name
    union all select 'humidity_pct'
    union all select 'wind_speed_ms'
    union all select 'precipitation_mm'
    union all select 'weather_code'
    union all select 'hourly_temperature_c'
    union all select 'hourly_humidity_pct'
    union all select 'hourly_wind_speed_ms'
    union all select 'hourly_precipitation_mm'
    union all select 'hourly_precipitation_probability_pct'
    union all select 'hourly_weather_code'
),
forbidden_columns as (
    select 'weather_main' as column_name
    union all select 'hourly_weather_main'
    union all select 'weather_description'
    union all select 'hourly_forecast_at'
    union all select 'source'
    union all select 'snapshot_bucket_at_utc'
    union all select 'temp_c'
    union all select 'precip_mm'
    union all select 'wind_kph'
    union all select 'rhum_pct'
    union all select 'pres_hpa'
    union all select 'wind_dir_deg'
    union all select 'wind_gust_kph'
    union all select 'snow_mm'
),
missing_required as (
    select rc.column_name
    from required_columns rc
    left join columns_present cp
        on rc.column_name = cp.column_name
    where cp.column_name is null
),
present_forbidden as (
    select fc.column_name
    from forbidden_columns fc
    inner join columns_present cp
        on fc.column_name = cp.column_name
)
select *
from missing_required
union all
select *
from present_forbidden
