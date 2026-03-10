select count(*) as current_station_count
from  {{ ref('dim_station') }}
where is_current = true

-- dbt run --select check_current_station_count