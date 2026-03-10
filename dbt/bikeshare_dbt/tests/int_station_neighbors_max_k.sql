{% set neighbor_k = var('station_neighbors_k', 5) | int %}

with violations as (
    select
        station_key,
        count(*) as neighbor_count
    from {{ ref('int_station_neighbors') }}
    group by station_key
    having count(*) > {{ neighbor_k }}
)
select *
from violations
