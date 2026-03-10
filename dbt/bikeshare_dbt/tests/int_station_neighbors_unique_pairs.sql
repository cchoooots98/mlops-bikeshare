with violations as (
    select
        station_key,
        neighbor_station_key,
        count(*) as row_count
    from {{ ref('int_station_neighbors') }}
    group by station_key, neighbor_station_key
    having count(*) > 1
)
select *
from violations
