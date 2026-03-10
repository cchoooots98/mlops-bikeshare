with expected_targets as (
    select
        cur.city,
        cur.station_id,
        cur.dt,
        fut.bikes as expected_target_bikes_t30,
        fut.docks as expected_target_docks_t30
    from {{ ref('feat_station_snapshot_5min') }} cur
    left join {{ ref('feat_station_snapshot_5min') }} fut
        on cur.city = fut.city
       and cur.station_id = fut.station_id
       and to_timestamp(fut.dt, 'YYYY-MM-DD-HH24-MI') = to_timestamp(cur.dt, 'YYYY-MM-DD-HH24-MI') + interval '30 minutes'
    where cur.target_bikes_t30 is not null
       or cur.target_docks_t30 is not null
)
select
    cur.city,
    cur.station_id,
    cur.dt,
    cur.target_bikes_t30,
    expected.expected_target_bikes_t30,
    cur.target_docks_t30,
    expected.expected_target_docks_t30
from {{ ref('feat_station_snapshot_5min') }} cur
inner join expected_targets expected
    on cur.city = expected.city
   and cur.station_id = expected.station_id
   and cur.dt = expected.dt
where coalesce(cur.target_bikes_t30, -1) <> coalesce(expected.expected_target_bikes_t30, -1)
   or coalesce(cur.target_docks_t30, -1) <> coalesce(expected.expected_target_docks_t30, -1)
