{{ config(tags=['deep_quality']) }}

{% set neighbor_k = var('station_neighbors_k', 5) | int %}
{% set radius_km = var('station_neighbors_radius_km', 0.8) | float %}
{% set earth_radius_km = 6371.0088 %}

with sampled_station_keys as (
    select station_key
    from {{ ref('dim_station') }}
    where is_current = true
      and latitude between -90 and 90
      and longitude between -180 and 180
    order by station_key
    limit 100
),
current_station_versions as (
    select
        station_key,
        city,
        latitude,
        longitude,
        radians(latitude) as latitude_rad,
        radians(longitude) as longitude_rad
    from {{ ref('dim_station') }}
    where is_current = true
      and latitude between -90 and 90
      and longitude between -180 and 180
),
exact_pairs as (
    select
        src.station_key,
        nbr.station_key as neighbor_station_key,
        {{ earth_radius_km }}::double precision * 2.0 * asin(
            sqrt(
                least(
                    1.0,
                    power(sin((nbr.latitude_rad - src.latitude_rad) / 2.0), 2)
                    + cos(src.latitude_rad)
                    * cos(nbr.latitude_rad)
                    * power(sin((nbr.longitude_rad - src.longitude_rad) / 2.0), 2)
                )
            )
        ) as distance_km
    from current_station_versions src
    inner join sampled_station_keys s
        on src.station_key = s.station_key
    inner join current_station_versions nbr
        on src.city = nbr.city
       and src.station_key <> nbr.station_key
),
exact_ranked as (
    select
        station_key,
        neighbor_station_key,
        row_number() over (
            partition by station_key
            order by distance_km, neighbor_station_key
        ) as neighbor_rank
    from exact_pairs
    where distance_km <= {{ radius_km }}::double precision
),
exact_selected as (
    select
        station_key,
        neighbor_station_key,
        neighbor_rank
    from exact_ranked
    where neighbor_rank <= {{ neighbor_k }}
),
model_selected as (
    select
        station_key,
        neighbor_station_key,
        neighbor_rank
    from {{ ref('int_station_neighbors') }}
    where station_key in (
        select station_key
        from sampled_station_keys
    )
),
differences as (
    select
        coalesce(e.station_key, m.station_key) as station_key,
        coalesce(e.neighbor_station_key, m.neighbor_station_key) as neighbor_station_key,
        e.neighbor_rank as expected_neighbor_rank,
        m.neighbor_rank as actual_neighbor_rank
    from exact_selected e
    full outer join model_selected m
        on e.station_key = m.station_key
       and e.neighbor_station_key = m.neighbor_station_key
       and e.neighbor_rank = m.neighbor_rank
    where e.station_key is null
       or m.station_key is null
)
select *
from differences
