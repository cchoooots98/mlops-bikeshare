{% test duplicate_grain(model, columns) %}
select
    {% for column in columns %}
    {{ column }}{% if not loop.last %},{% endif %}
    {% endfor %},
    count(*) as row_count
from {{ model }}
group by
    {% for column in columns %}
    {{ column }}{% if not loop.last %},{% endif %}
    {% endfor %}
having count(*) > 1
{% endtest %}
