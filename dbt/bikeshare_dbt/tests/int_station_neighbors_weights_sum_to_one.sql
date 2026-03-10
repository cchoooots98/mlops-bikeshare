with violations as (
    select
        station_key,
        sum(neighbor_weight) as weight_sum
    from {{ ref('int_station_neighbors') }}
    group by station_key
    having abs(sum(neighbor_weight) - 1.0) > 0.000001
)
select *
from violations
