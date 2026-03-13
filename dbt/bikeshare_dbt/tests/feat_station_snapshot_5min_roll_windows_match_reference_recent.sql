{{ config(tags=['quality_gate']) }}

{% set snapshot_step_minutes = var('feature_snapshot_step_minutes', 5) | int %}
{% set max_roll_window_minutes = var('feature_max_roll_window_minutes', 60) | int %}
{% set output_window_minutes = var('feature_rebuild_lookback_minutes', 30) | int %}
{% set comparison_tolerance = 0.000001 %}

with comparison_window as (
    select
        {{ runtime_utc_expr('test_window_end_utc') }} as window_end_utc,
        {{ runtime_utc_expr('test_window_end_utc') }}
            - interval '{{ output_window_minutes }} minutes' as output_start_utc,
        {{ runtime_utc_expr('test_window_end_utc') }}
            - interval '{{ output_window_minutes + max_roll_window_minutes }} minutes' as source_start_utc
),
source_rows as (
    select
        e.city,
        e.station_id,
        e.snapshot_bucket_at_utc,
        e.num_bikes_available as bikes,
        e.num_docks_available as docks,
        e.capacity,
        e.minutes_since_prev_snapshot,
        e.prev_num_bikes_available,
        e.prev_num_docks_available
    from {{ ref('int_station_status_enriched') }} e
    cross join comparison_window cw
    where e.snapshot_bucket_at_utc >= cw.source_start_utc
      and e.snapshot_bucket_at_utc < cw.window_end_utc
),
station_features as (
    select
        city,
        station_id,
        snapshot_bucket_at_utc,
        bikes::double precision as bikes,
        case
            when coalesce(minutes_since_prev_snapshot, 0.0) > 0.0
                then coalesce((bikes - prev_num_bikes_available)::double precision, 0.0)
                    / minutes_since_prev_snapshot::double precision
                    * {{ snapshot_step_minutes }}::double precision
            else 0.0
        end as delta_bikes_5m
    from source_rows
),
reference_rolls as (
    select
        city,
        station_id,
        snapshot_bucket_at_utc,
        sum(delta_bikes_5m) over (
            partition by city, station_id
            order by snapshot_bucket_at_utc
            range between interval '15 minutes' preceding and current row
        ) as expected_roll15_net_bikes,
        sum(delta_bikes_5m) over (
            partition by city, station_id
            order by snapshot_bucket_at_utc
            range between interval '30 minutes' preceding and current row
        ) as expected_roll30_net_bikes,
        sum(delta_bikes_5m) over (
            partition by city, station_id
            order by snapshot_bucket_at_utc
            range between interval '{{ max_roll_window_minutes }} minutes' preceding and current row
        ) as expected_roll60_net_bikes,
        avg(bikes) over (
            partition by city, station_id
            order by snapshot_bucket_at_utc
            range between interval '15 minutes' preceding and current row
        ) as expected_roll15_bikes_mean,
        avg(bikes) over (
            partition by city, station_id
            order by snapshot_bucket_at_utc
            range between interval '30 minutes' preceding and current row
        ) as expected_roll30_bikes_mean,
        avg(bikes) over (
            partition by city, station_id
            order by snapshot_bucket_at_utc
            range between interval '{{ max_roll_window_minutes }} minutes' preceding and current row
        ) as expected_roll60_bikes_mean
    from station_features
),
actual_rolls as (
    select
        f.city,
        f.station_id,
        {{ feature_dt_to_utc('f.dt') }} as snapshot_bucket_at_utc,
        f.roll15_net_bikes,
        f.roll30_net_bikes,
        f.roll60_net_bikes,
        f.roll15_bikes_mean,
        f.roll30_bikes_mean,
        f.roll60_bikes_mean
    from {{ ref('feat_station_snapshot_5min') }} f
    cross join comparison_window cw
    where {{ feature_dt_to_utc('f.dt') }} >= cw.output_start_utc
      and {{ feature_dt_to_utc('f.dt') }} < cw.window_end_utc
),
violations as (
    select
        a.city,
        a.station_id,
        a.snapshot_bucket_at_utc,
        a.roll15_net_bikes,
        r.expected_roll15_net_bikes,
        a.roll30_net_bikes,
        r.expected_roll30_net_bikes,
        a.roll60_net_bikes,
        r.expected_roll60_net_bikes,
        a.roll15_bikes_mean,
        r.expected_roll15_bikes_mean,
        a.roll30_bikes_mean,
        r.expected_roll30_bikes_mean,
        a.roll60_bikes_mean,
        r.expected_roll60_bikes_mean
    from actual_rolls a
    inner join reference_rolls r
        on a.city = r.city
       and a.station_id = r.station_id
       and a.snapshot_bucket_at_utc = r.snapshot_bucket_at_utc
)
select *
from violations
where abs(coalesce(roll15_net_bikes, 0.0) - coalesce(expected_roll15_net_bikes, 0.0)) > {{ comparison_tolerance }}
   or abs(coalesce(roll30_net_bikes, 0.0) - coalesce(expected_roll30_net_bikes, 0.0)) > {{ comparison_tolerance }}
   or abs(coalesce(roll60_net_bikes, 0.0) - coalesce(expected_roll60_net_bikes, 0.0)) > {{ comparison_tolerance }}
   or abs(coalesce(roll15_bikes_mean, 0.0) - coalesce(expected_roll15_bikes_mean, 0.0)) > {{ comparison_tolerance }}
   or abs(coalesce(roll30_bikes_mean, 0.0) - coalesce(expected_roll30_bikes_mean, 0.0)) > {{ comparison_tolerance }}
   or abs(coalesce(roll60_bikes_mean, 0.0) - coalesce(expected_roll60_bikes_mean, 0.0)) > {{ comparison_tolerance }}
