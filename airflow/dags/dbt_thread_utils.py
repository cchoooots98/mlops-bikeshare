from __future__ import annotations


def get_dbt_threads(get_setting, lane_key: str, *, default_threads: str = "2") -> int:
    shared_default = get_setting("DBT_THREADS", "DBT_THREADS", default_threads)
    return int(get_setting(lane_key, lane_key, shared_default))
