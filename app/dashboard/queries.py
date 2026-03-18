"""PostgreSQL query helpers for the dashboard.

Replaces the former Athena-based SQL functions.
Reads station metadata and data-freshness from the analytics schema.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

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
            CAST(station_id AS text)            AS station_id,
            MAX(name)                           AS name,
            MAX(capacity)                       AS capacity,
            AVG(lat)                            AS lat,
            AVG(lon)                            AS lon
        FROM {schema}.feat_station_snapshot_latest
        WHERE city = :city
        GROUP BY station_id
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"city": city})
    return df


def load_freshness(*, engine: Engine, schema: str, city: str, tables: list[str]) -> pd.DataFrame:
    """Return the latest dt string and computed delay for each monitored table.

    Returns DataFrame: source (str), latest_dt_str (str or None).
    """
    schema = validate_pg_identifier(schema)
    rows = []
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
        except Exception:
            latest = None
        rows.append({"source": table, "latest_dt_str": latest})
    return pd.DataFrame(rows)
