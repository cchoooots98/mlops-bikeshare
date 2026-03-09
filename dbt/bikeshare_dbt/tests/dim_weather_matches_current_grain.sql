select
    c.weather_current_pk
from {{ ref('stg_weather_current') }} c
left join {{ ref('dim_weather') }} d
    on d.weather_key = c.weather_current_pk
where d.weather_key is null
