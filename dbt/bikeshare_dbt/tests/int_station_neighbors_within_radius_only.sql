{% set radius_km = var('station_neighbors_radius_km', 0.8) | float %}

select
    station_key,
    neighbor_station_key,
    distance_km
from {{ ref('int_station_neighbors') }}
where distance_km > {{ radius_km }}::double precision
