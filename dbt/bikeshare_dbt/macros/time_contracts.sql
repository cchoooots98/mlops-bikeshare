{% macro feature_dt_from_utc(ts_expr) -%}
to_char({{ ts_expr }} at time zone 'UTC', 'YYYY-MM-DD-HH24-MI')
{%- endmacro %}

{% macro feature_dt_to_utc(dt_expr) -%}
(to_timestamp({{ dt_expr }}, 'YYYY-MM-DD-HH24-MI')::timestamp at time zone 'UTC')
{%- endmacro %}

{% macro utc_ts_key(ts_expr) -%}
to_char({{ ts_expr }}, 'YYYY-MM-DD HH24:MI:SSOF')
{%- endmacro %}
