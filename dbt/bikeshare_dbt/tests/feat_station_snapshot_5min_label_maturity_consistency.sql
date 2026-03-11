select
    city,
    station_id,
    dt,
    target_bikes_t30,
    target_docks_t30,
    y_stockout_bikes_30,
    y_stockout_docks_30
from {{ ref('feat_station_snapshot_5min') }}
where not (
        y_stockout_bikes_30 is null
    and y_stockout_docks_30 is null
    and target_bikes_t30 is null
    and target_docks_t30 is null
)
and (
        y_stockout_bikes_30 is null
     or y_stockout_docks_30 is null
     or target_bikes_t30 is null
     or target_docks_t30 is null
)
