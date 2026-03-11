with overlapping_rows as (
    select
        f.city,
        f.station_id,
        f.dt,
        a.snapshot_bucket_at_utc
    from {{ ref('feat_station_snapshot_5min') }} f
    inner join (
        select
            fs.city,
            fs.station_id,
            fs.snapshot_bucket_at_utc
        from {{ ref('fct_station_status') }} fs
        inner join {{ ref('dim_station') }} ds
            on fs.station_version_key = ds.station_version_key
        where fs.num_bikes_available + fs.num_docks_available > ds.capacity
    ) a
        on f.city = a.city
       and f.station_id = a.station_id
       and {{ feature_dt_to_utc('f.dt') }} = a.snapshot_bucket_at_utc
)
select *
from overlapping_rows
