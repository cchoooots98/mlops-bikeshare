{{ config(tags=['quality_gate']) }}

with expected as (
    select
        cur.city,
        cur.station_id,
        cur.dt,
        case
            when min(fut.bikes) <= {{ var('stockout_threshold', 2) | int }} then 1
            else 0
        end as expected_bike_label,
        case
            when min(fut.docks) <= {{ var('stockout_threshold', 2) | int }} then 1
            else 0
        end as expected_dock_label
    from {{ ref('feat_station_snapshot_5min') }} cur
    left join {{ ref('feat_station_snapshot_5min') }} fut
        on cur.city = fut.city
       and cur.station_id = fut.station_id
       and {{ feature_dt_to_utc('fut.dt') }} > {{ feature_dt_to_utc('cur.dt') }}
       and {{ feature_dt_to_utc('fut.dt') }} <= {{ feature_dt_to_utc('cur.dt') }} + interval '30 minutes'
    where cur.y_stockout_bikes_30 is not null
      and cur.y_stockout_docks_30 is not null
    group by cur.city, cur.station_id, cur.dt
)
select
    cur.city,
    cur.station_id,
    cur.dt,
    cur.y_stockout_bikes_30,
    expected.expected_bike_label,
    cur.y_stockout_docks_30,
    expected.expected_dock_label
from {{ ref('feat_station_snapshot_5min') }} cur
inner join expected
    on cur.city = expected.city
   and cur.station_id = expected.station_id
   and cur.dt = expected.dt
where cur.y_stockout_bikes_30 <> expected.expected_bike_label
   or cur.y_stockout_docks_30 <> expected.expected_dock_label
