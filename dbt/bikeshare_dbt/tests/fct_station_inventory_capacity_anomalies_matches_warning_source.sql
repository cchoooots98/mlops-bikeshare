{{ config(tags=['quality_gate']) }}

with warning_source as (
    select
        f.city,
        f.station_id,
        f.station_version_key,
        f.snapshot_bucket_at_utc,
        f.num_bikes_available,
        f.num_docks_available,
        d.capacity,
        (f.num_bikes_available + f.num_docks_available - d.capacity) as over_capacity_by
    from {{ ref('fct_station_status') }} f
    inner join {{ ref('dim_station') }} d
        on f.station_version_key = d.station_version_key
    where f.num_bikes_available + f.num_docks_available > d.capacity
),
anomaly_view as (
    select
        city,
        station_id,
        station_version_key,
        snapshot_bucket_at_utc,
        num_bikes_available,
        num_docks_available,
        capacity,
        over_capacity_by
    from {{ ref('fct_station_inventory_capacity_anomalies') }}
),
only_in_warning_source as (
    select *
    from warning_source
    except
    select *
    from anomaly_view
),
only_in_anomaly_view as (
    select *
    from anomaly_view
    except
    select *
    from warning_source
)
select *
from only_in_warning_source
union all
select *
from only_in_anomaly_view
