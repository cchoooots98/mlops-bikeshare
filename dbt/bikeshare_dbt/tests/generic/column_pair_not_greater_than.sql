{% test column_pair_not_greater_than(model, column_name, other_column) %}

select *
from {{ model }}
where {{ column_name }} is not null
  and {{ other_column }} is not null
  and {{ column_name }} > {{ other_column }}

{% endtest %}
