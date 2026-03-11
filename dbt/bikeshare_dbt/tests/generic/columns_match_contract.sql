{% test columns_match_contract(model, required_columns=[], forbidden_columns=[]) %}
with columns_present as (
    select lower(column_name) as column_name
    from information_schema.columns
    where table_schema = '{{ model.schema }}'
      and table_name = '{{ model.identifier }}'
),
required_columns as (
    {% if required_columns | length == 0 %}
    select null::text as column_name
    where false
    {% else %}
    {% for column_name in required_columns %}
    select '{{ column_name | lower }}' as column_name{% if not loop.last %} union all{% endif %}
    {% endfor %}
    {% endif %}
),
forbidden_columns as (
    {% if forbidden_columns | length == 0 %}
    select null::text as column_name
    where false
    {% else %}
    {% for column_name in forbidden_columns %}
    select '{{ column_name | lower }}' as column_name{% if not loop.last %} union all{% endif %}
    {% endfor %}
    {% endif %}
),
missing_required as (
    select rc.column_name
    from required_columns rc
    left join columns_present cp
        on rc.column_name = cp.column_name
    where cp.column_name is null
),
present_forbidden as (
    select fc.column_name
    from forbidden_columns fc
    inner join columns_present cp
        on fc.column_name = cp.column_name
)
select *
from missing_required
union all
select *
from present_forbidden
{% endtest %}
