{% macro station_inventory_capacity_multiplier() -%}
{{ return(var('station_inventory_capacity_multiplier', 2) | float) }}
{%- endmacro %}

{% macro station_inventory_limit_expr(capacity_column) -%}
({{ capacity_column }}::double precision * {{ station_inventory_capacity_multiplier() }})
{%- endmacro %}

{% macro station_inventory_within_limit_expr(bikes_column, docks_column, capacity_column) -%}
(
    coalesce({{ bikes_column }}, 0)::double precision
    + coalesce({{ docks_column }}, 0)::double precision
    <= {{ station_inventory_limit_expr(capacity_column) }}
)
{%- endmacro %}
