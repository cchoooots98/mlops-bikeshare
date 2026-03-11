{% macro station_snapshot_key(city_expr, station_id_expr, ts_expr) -%}
concat(
    {{ city_expr }},
    '|',
    {{ utc_ts_key(ts_expr) }},
    '|',
    {{ station_id_expr }}
)
{%- endmacro %}
