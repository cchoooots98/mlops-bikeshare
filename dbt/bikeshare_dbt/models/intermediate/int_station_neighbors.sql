{{ config(
    materialized='table'
) }}

{% set neighbor_k = var('station_neighbors_k', 5) | int %}
{% set radius_km = var('station_neighbors_radius_km', 0.8) | float %}
{% set earth_radius_km = 6371.0088 %}
{% set distance_eps_km = 0.000001 %}
{% set km_per_degree_lat = 111.32 %}

with current_station_versions as (
    select
        station_key,
        city,
        station_id,
        latitude,
        longitude,
        radians(latitude) as latitude_rad,
        radians(longitude) as longitude_rad,
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
candidate_pairs as (
    select
        src.city,
        src.station_key,
        src.station_id,
        src.latitude,
        src.longitude,
        src.latitude_rad,
        src.longitude_rad,
        nbr.station_key as neighbor_station_key,
        nbr.station_id as neighbor_station_id,
        nbr.latitude as neighbor_latitude,
        nbr.longitude as neighbor_longitude,
        nbr.latitude_rad as neighbor_latitude_rad,
        nbr.longitude_rad as neighbor_longitude_rad
    from current_station_versions src
    inner join current_station_versions nbr
        on src.city = nbr.city
       and src.station_key <> nbr.station_key
       and abs(src.latitude - nbr.latitude) <= src.latitude_delta_deg
       and abs(src.longitude - nbr.longitude) <= src.longitude_delta_deg
),
station_pairs as (
    select
        city,
        station_key,
        station_id,
        neighbor_station_key,
        neighbor_station_id,
        {{ earth_radius_km }}::double precision * 2.0 * asin(
            sqrt(
                least(
                    1.0,
                    power(sin((neighbor_latitude_rad - latitude_rad) / 2.0), 2)
                    + cos(latitude_rad)
                    * cos(neighbor_latitude_rad)
                    * power(sin((neighbor_longitude_rad - longitude_rad) / 2.0), 2)
                )
            )
        ) as distance_km
    from candidate_pairs
),
within_radius_pairs as (
    select
        city,
        station_key,
        station_id,
        neighbor_station_key,
        neighbor_station_id,
        distance_km
    from station_pairs
    where distance_km <= {{ radius_km }}::double precision
),
ranked_pairs as (
    select
        city,
        station_key,
        station_id,
        neighbor_station_key,
        neighbor_station_id,
        distance_km,
        count(*) over (
            partition by station_key
        ) as neighbor_count_within_radius,
        row_number() over (
            partition by station_key
            order by distance_km, neighbor_station_key
        ) as neighbor_rank
    from within_radius_pairs
),
selected_neighbors as (
    select
        city,
        station_key,
        station_id,
        neighbor_station_key,
        neighbor_station_id,
        distance_km,
        neighbor_count_within_radius,
        neighbor_rank
    from ranked_pairs
    where neighbor_rank <= {{ neighbor_k }}
),
weighted_neighbors as (
    select
        city,
        station_key,
        station_id,
        neighbor_station_key,
        neighbor_station_id,
        distance_km,
        neighbor_count_within_radius,
        neighbor_rank,
        1.0 / (distance_km + {{ distance_eps_km }}::double precision) as inverse_distance_weight
    from selected_neighbors
)
select
    city,
    station_key,
    station_id,
    neighbor_station_key,
    neighbor_station_id,
    distance_km,
    neighbor_count_within_radius,
    neighbor_rank,
    inverse_distance_weight
    / nullif(
        sum(inverse_distance_weight) over (
            partition by station_key
        ),
        0.0
    ) as neighbor_weight
from weighted_neighbors
