{% macro runtime_utc_expr(var_name, default_sql='current_timestamp') -%}
    {%- set runtime_value = var(var_name, none) -%}
    {%- if runtime_value is not none and runtime_value | string | trim != '' -%}
        '{{ runtime_value }}'::timestamptz
    {%- else -%}
        {{ default_sql }}
    {%- endif -%}
{%- endmacro %}

{% macro runtime_window_start_utc_expr(end_var_name='test_window_end_utc', lookback_var_name='test_window_lookback_hours', default_lookback_hours=72) -%}
    {{ runtime_utc_expr(end_var_name) }} - interval '{{ var(lookback_var_name, default_lookback_hours) | int }} hours'
{%- endmacro %}
