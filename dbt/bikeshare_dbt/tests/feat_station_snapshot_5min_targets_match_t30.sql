{{ config(tags=['quality_gate']) }}

with expected_targets as (
    select
        cur.city,
        cur.station_id,
        cur.dt,
        max({{ feature_dt_to_utc('fut.dt') }}) as expected_target_snapshot_bucket_at_utc
    from {{ ref('feat_station_snapshot_5min') }} cur
    left join {{ ref('feat_station_snapshot_5min') }} fut
        on cur.city = fut.city
       and cur.station_id = fut.station_id
       and {{ feature_dt_to_utc('fut.dt') }} > {{ feature_dt_to_utc('cur.dt') }}
       and {{ feature_dt_to_utc('fut.dt') }} <= {{ feature_dt_to_utc('cur.dt') }} + interval '30 minutes'
    where cur.target_bikes_t30 is not null
       or cur.target_docks_t30 is not null
    group by cur.city, cur.station_id, cur.dt
),
expected_target_values as (
    select
        expected.city,
        expected.station_id,
        expected.dt,
        fut.bikes as expected_target_bikes_t30,
        fut.docks as expected_target_docks_t30
    from expected_targets expected
    left join {{ ref('feat_station_snapshot_5min') }} fut
        on expected.city = fut.city
       and expected.station_id = fut.station_id
       and expected.expected_target_snapshot_bucket_at_utc = {{ feature_dt_to_utc('fut.dt') }}
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
inner join expected_target_values expected
    on cur.city = expected.city
   and cur.station_id = expected.station_id
   and cur.dt = expected.dt
where coalesce(cur.target_bikes_t30, -1) <> coalesce(expected.expected_target_bikes_t30, -1)
   or coalesce(cur.target_docks_t30, -1) <> coalesce(expected.expected_target_docks_t30, -1)
