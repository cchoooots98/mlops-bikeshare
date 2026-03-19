{{ config(tags=['quality_gate']) }}

{% set snapshot_step_minutes = var('feature_snapshot_step_minutes', 5) | int %}
{% set label_horizon_minutes = var('feature_label_horizon_minutes', 30) | int %}
{% set immature_window_minutes = label_horizon_minutes %}

with city_latest as (
    select
        city,
        max({{ feature_dt_to_utc('dt') }}) + interval '5 minutes' as latest_dt
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
where {{ feature_dt_to_utc('f.dt') }} > cl.latest_dt - interval '{{ immature_window_minutes }} minutes'
  and (
    f.y_stockout_bikes_30 is not null
    or f.target_bikes_t30 is not null
    or f.y_stockout_docks_30 is not null
    or f.target_docks_t30 is not null
  )
