with violations as (
    select
        latest.city,
        latest.station_id,
        latest.dt,
        max(hist.dt) as observed_max_dt
    from {{ ref('feat_station_snapshot_latest') }} latest
    inner join {{ ref('feat_station_snapshot_5min') }} hist
        on latest.city = hist.city
       and latest.station_id = hist.station_id
    group by latest.city, latest.station_id, latest.dt
    having max(hist.dt) <> latest.dt
)
select *
from violations
