from __future__ import annotations

from .targeting import DashboardTargetConfig


def build_station_info_query(*, database: str, view_name: str, city: str) -> str:
    return f"""
WITH ranked AS (
    SELECT
        station_id,
        name,
        capacity,
        lat,
        lon,
        dt_ts,
        row_number() OVER (PARTITION BY station_id ORDER BY dt_ts DESC) AS rn
    FROM {database}.{view_name}
    WHERE city = '{city}'
)
SELECT station_id, name, capacity, lat, lon
FROM ranked
WHERE rn = 1
"""


def build_latest_predictions_query(*, database: str, view_name: str, city: str, target: DashboardTargetConfig) -> str:
    return f"""
WITH ranked AS (
    SELECT
        CAST(station_id AS varchar) AS station_id,
        TRY(date_parse(dt, '%%Y-%%m-%%d-%%H-%%i')) AS ts,
        CAST({target.score_column} AS double) AS score,
        row_number() OVER (
            PARTITION BY CAST(station_id AS varchar)
            ORDER BY TRY(date_parse(dt, '%%Y-%%m-%%d-%%H-%%i')) DESC
        ) AS rn
    FROM {database}.{view_name}
    WHERE city = '{city}'
      AND prediction_target = '{target.target_name}'
      AND {target.score_column} IS NOT NULL
)
SELECT station_id, ts, score
FROM ranked
WHERE rn = 1
"""


def build_prediction_history_query(
    *,
    database: str,
    view_name: str,
    city: str,
    station_id: str,
    target: DashboardTargetConfig,
    limit: int,
) -> str:
    return f"""
SELECT
    CAST(station_id AS varchar) AS station_id,
    TRY(date_parse(dt, '%Y-%m-%d-%H-%i')) AS ts,
    CAST({target.score_column} AS double) AS score
FROM {database}.{view_name}
WHERE city = '{city}'
  AND prediction_target = '{target.target_name}'
  AND CAST(station_id AS varchar) = '{station_id}'
  AND {target.score_column} IS NOT NULL
  AND TRY(date_parse(dt, '%Y-%m-%d-%H-%i')) IS NOT NULL
ORDER BY ts DESC
LIMIT {limit}
"""


def build_quality_summary_query(*, database: str, view_name: str, city: str, target: DashboardTargetConfig) -> str:
    return f"""
SELECT
    dt,
    CAST({target.score_column} AS double) AS score,
    CAST({target.label_column} AS integer) AS label
FROM {database}.{view_name}
WHERE city = '{city}'
  AND prediction_target = '{target.target_name}'
  AND {target.score_column} IS NOT NULL
  AND {target.label_column} IS NOT NULL
"""


def build_freshness_query(*, database: str, table_name: str, city: str) -> str:
    return f"""
SELECT '{table_name}' AS source, max(dt) AS latest_dt_str
FROM {database}.{table_name}
WHERE city = '{city}'
"""
