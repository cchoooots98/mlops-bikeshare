{{ config(tags=['deep_quality']) }}

select
    station_key
from {{ ref('dim_station') }}
group by station_key
having sum(case when is_current then 1 else 0 end) <> 1
