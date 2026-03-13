{{ config(tags=['quality_gate']) }}

with source_feature_rows as (
    select
        city,
        station_id,
        snapshot_bucket_at_utc,
        num_bikes_available as bikes,
        num_docks_available as docks
    from {{ ref('int_station_status_enriched') }}
    where snapshot_bucket_at_utc >= {{ runtime_window_start_utc_expr(default_lookback_hours=72) }}
      and snapshot_bucket_at_utc < {{ runtime_utc_expr('test_window_end_utc') }}
),
neighbor_observations as (
    select
        cur.city,
        cur.station_id,
        cur.snapshot_bucket_at_utc,
        count(*) filter (where nbr.station_id is not null) as observed_neighbor_rows
    from source_feature_rows cur
    inner join {{ ref('int_station_neighbors') }} n
        on cur.city = n.city
       and cur.station_id = n.station_id
    left join source_feature_rows nbr
        on nbr.city = n.city
       and nbr.station_id = n.neighbor_station_id
       and nbr.snapshot_bucket_at_utc = cur.snapshot_bucket_at_utc
    group by cur.city, cur.station_id, cur.snapshot_bucket_at_utc
),
violations as (
    select
        f.city,
        f.station_id,
        f.dt,
        n.observed_neighbor_rows,
        f.nbr_bikes_weighted,
        f.nbr_docks_weighted,
        f.has_neighbors_within_radius,
        f.neighbor_count_within_radius
    from {{ ref('feat_station_snapshot_5min') }} f
    inner join neighbor_observations n
        on f.city = n.city
       and f.station_id = n.station_id
       and {{ feature_dt_to_utc('f.dt') }} = n.snapshot_bucket_at_utc
    where n.observed_neighbor_rows > 0
      and (
            f.nbr_bikes_weighted is null
         or f.nbr_docks_weighted is null
         or f.has_neighbors_within_radius <> 1
         or f.neighbor_count_within_radius <= 0
      )
)
select *
from violations
