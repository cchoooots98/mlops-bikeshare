select *
from {{ ref('stg_station_status') }}
where num_bikes_available < 0
   or num_docks_available < 0
