with city_latest as (
    select
        city,
        max(to_timestamp(dt, 'YYYY-MM-DD-HH24-MI')) as latest_dt
    from {{ ref('feat_station_snapshot_5min') }}
    group by city
)
select
    f.city,
    f.station_id,
    f.dt,
    f.y_stockout_bikes_30,
    f.y_stockout_docks_30
from {{ ref('feat_station_snapshot_5min') }} f
inner join city_latest cl
    on f.city = cl.city
where to_timestamp(f.dt, 'YYYY-MM-DD-HH24-MI') > cl.latest_dt - interval '30 minutes'
  and (
    f.y_stockout_bikes_30 is not null
    or f.y_stockout_docks_30 is not null
  )
