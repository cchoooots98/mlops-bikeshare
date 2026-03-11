select
    city,
    station_id,
    snapshot_bucket_at_utc,
    over_capacity_by
from {{ ref('fct_station_inventory_capacity_anomalies') }}
where over_capacity_by <= 0
