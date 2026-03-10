with stations_without_neighbors as (
    select
        f.city,
        f.station_id,
        f.dt,
        f.nbr_bikes_weighted,
        f.nbr_docks_weighted,
        f.has_neighbors_within_radius,
        f.neighbor_count_within_radius
    from {{ ref('feat_station_snapshot_5min') }} f
    left join (
        select distinct city, station_id
        from {{ ref('int_station_neighbors') }}
    ) n
        on f.city = n.city
       and f.station_id = n.station_id
    where n.station_id is null
)
select *
from stations_without_neighbors
where nbr_bikes_weighted <> 0.0
   or nbr_docks_weighted <> 0.0
   or has_neighbors_within_radius <> 0
   or neighbor_count_within_radius <> 0
