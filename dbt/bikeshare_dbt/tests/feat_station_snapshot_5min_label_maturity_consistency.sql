{{ config(tags=['quality_gate']) }}

with feature_rows as (
    select
        city,
        station_id,
        dt,
        {{ feature_dt_to_utc('dt') }} as snapshot_bucket_at_utc,
        target_bikes_t30,
        target_docks_t30,
        y_stockout_bikes_30,
        y_stockout_docks_30
    from {{ ref('feat_station_snapshot_5min') }}
),
city_latest as (
    select
        city,
        max(snapshot_bucket_at_utc) + interval '5 minutes' as latest_snapshot_bucket_at_utc
    from feature_rows
    group by city
),
eligible_rows as (
    select
        f.city,
        f.station_id,
        f.dt,
        f.target_bikes_t30,
        f.target_docks_t30,
        f.y_stockout_bikes_30,
        f.y_stockout_docks_30
    from feature_rows f
    inner join city_latest cl
        on f.city = cl.city
    where cl.latest_snapshot_bucket_at_utc >= f.snapshot_bucket_at_utc + interval '30 minutes'
      and exists (
          select 1
          from {{ ref('int_station_status_enriched') }} fut
          where fut.city = f.city
            and fut.station_id = f.station_id
            and fut.snapshot_bucket_at_utc > f.snapshot_bucket_at_utc
            and fut.snapshot_bucket_at_utc <= f.snapshot_bucket_at_utc + interval '30 minutes'
            and fut.snapshot_bucket_at_utc <= cl.latest_snapshot_bucket_at_utc
      )
),
partial_null_rows as (
    select
        city,
        station_id,
        dt,
        target_bikes_t30,
        target_docks_t30,
        y_stockout_bikes_30,
        y_stockout_docks_30
    from feature_rows
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
),
eligible_but_null as (
    select *
    from eligible_rows
    where y_stockout_bikes_30 is null
       or y_stockout_docks_30 is null
       or target_bikes_t30 is null
       or target_docks_t30 is null
)
select *
from partial_null_rows

union all

select *
from eligible_but_null
