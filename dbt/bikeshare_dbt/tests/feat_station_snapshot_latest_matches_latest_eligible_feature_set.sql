{{ config(tags=['quality_gate']) }}


with recent_source_rows as (
    select
        city,
        station_id,
        snapshot_bucket_at_utc,
        row_number() over (
            partition by city, station_id
            order by snapshot_bucket_at_utc desc
        ) as row_num
    from {{ ref('int_station_status_enriched') }}
    where snapshot_bucket_at_utc >= {{ runtime_window_start_utc_expr(default_lookback_hours=72) }}
      and snapshot_bucket_at_utc < {{ runtime_utc_expr('test_window_end_utc') }}
),
expected_latest as (
    select
        city,
        station_id,
        {{ feature_dt_from_utc('snapshot_bucket_at_utc') }} as expected_dt
    from recent_source_rows
    where row_num = 1
),
actual_latest as (
    select
        city,
        station_id,
        dt
    from {{ ref('feat_station_snapshot_latest') }}
),
missing_or_stale as (
    select
        e.city,
        e.station_id,
        e.expected_dt,
        a.dt as actual_dt,
        case
            when a.station_id is null then 'missing_from_latest'
            else 'latest_dt_stale'
        end as mismatch_type
    from expected_latest e
    left join actual_latest a
        on e.city = a.city
       and e.station_id = a.station_id
    where a.station_id is null
       or {{ feature_dt_to_utc('a.dt') }} < {{ feature_dt_to_utc('e.expected_dt') }}
)
select *
from missing_or_stale
