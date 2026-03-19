"""PostgreSQL query helpers for the dashboard.

Replaces the former Athena-based SQL functions.
Reads station metadata and data-freshness from the analytics schema.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from dashboard.contracts import FreshnessLoadResult, LoadStatus
from dashboard.utils import validate_pg_identifier


def load_station_info(*, engine: Engine, schema: str, city: str) -> pd.DataFrame:
    """Return one row per station with its latest lat/lon/name/capacity.

    Queries feat_station_snapshot_latest which holds the most recent
    snapshot per station — a lightweight dedup via GROUP BY.

    Returns DataFrame: station_id (str), name, capacity, lat, lon.
    """
    schema = validate_pg_identifier(schema)
    sql = text(f"""
        SELECT
            CAST(f.station_id AS text)          AS station_id,
            COALESCE(ds.station_name, CAST(f.station_id AS text)) AS name,
            COALESCE(ds.capacity, f.capacity)   AS capacity,
            COALESCE(ds.latitude, f.lat)        AS lat,
            COALESCE(ds.longitude, f.lon)       AS lon
        FROM {schema}.feat_station_snapshot_latest f
        LEFT JOIN {schema}.dim_station ds
            ON f.city = ds.city
           AND CAST(f.station_id AS text) = CAST(ds.station_id AS text)
           AND ds.is_current = TRUE
        WHERE f.city = :city
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"city": city})
    return df


def load_freshness(*, engine: Engine, schema: str, city: str, tables: list[str]) -> FreshnessLoadResult:
    """Return the latest dt string and computed delay for each monitored table.

    Returns DataFrame: source (str), latest_dt_str (str or None).
    """
    schema = validate_pg_identifier(schema)
    rows = []
    overall_status = LoadStatus.OK
    overall_messages: list[str] = []
    for table in tables:
        table = validate_pg_identifier(table)
        sql = text(f"""
            SELECT MAX(dt) AS latest_dt_str
            FROM {schema}.{table}
            WHERE city = :city
        """)
        try:
            with engine.connect() as conn:
                result = conn.execute(sql, {"city": city}).fetchone()
            latest = result[0] if result else None
            row_status = "ok"
            row_message = ""
        except Exception as exc:
            latest = None
            row_status = LoadStatus.READ_ERROR.value
            row_message = str(exc)
            overall_status = LoadStatus.READ_ERROR
            overall_messages.append(f"{table}: {exc}")
        rows.append(
            {
                "source": table,
                "latest_dt_str": latest,
                "loader_status": row_status,
                "message": row_message,
            }
        )
    return FreshnessLoadResult(
        status=overall_status,
        data=pd.DataFrame(rows),
        message="; ".join(overall_messages),
    )
