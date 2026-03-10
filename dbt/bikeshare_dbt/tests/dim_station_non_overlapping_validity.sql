select
    left_side.station_version_key as left_station_version_key,
    right_side.station_version_key as right_station_version_key
from {{ ref('dim_station') }} left_side
join {{ ref('dim_station') }} right_side
    on left_side.station_key = right_side.station_key
   and left_side.station_version_key < right_side.station_version_key
   and left_side.valid_from_utc < coalesce(right_side.valid_to_utc, 'infinity'::timestamptz)
   and right_side.valid_from_utc < coalesce(left_side.valid_to_utc, 'infinity'::timestamptz)
