{{ config(tags=['deep_quality']) }}

{% set radius_km = var('station_neighbors_radius_km', 0.8) | float %}
{% set km_per_degree_lat = 111.32 %}

with current_station_versions as (
    select
        station_key,
        city,
        latitude,
        longitude,
        radians(latitude) as latitude_rad,
        {{ radius_km }}::double precision / {{ km_per_degree_lat }}::double precision as latitude_delta_deg,
        {{ radius_km }}::double precision / (
            {{ km_per_degree_lat }}::double precision
            * greatest(abs(cos(radians(latitude))), 0.01)
        ) as longitude_delta_deg
    from {{ ref('dim_station') }}
    where is_current = true
      and latitude between -90 and 90
      and longitude between -180 and 180
),
full_pair_count as (
    select count(*) as pair_count
    from current_station_versions src
    inner join current_station_versions nbr
        on src.city = nbr.city
       and src.station_key <> nbr.station_key
),
bbox_candidate_count as (
    select count(*) as pair_count
    from current_station_versions src
    inner join current_station_versions nbr
        on src.city = nbr.city
       and src.station_key <> nbr.station_key
       and abs(src.latitude - nbr.latitude) <= src.latitude_delta_deg
       and abs(src.longitude - nbr.longitude) <= src.longitude_delta_deg
)
select
    full_pair_count.pair_count as full_pair_count,
    bbox_candidate_count.pair_count as bbox_candidate_count
from full_pair_count
cross join bbox_candidate_count
where bbox_candidate_count.pair_count >= full_pair_count.pair_count
