import argparse
import json
import os
import time
from datetime import date, datetime, timezone

import pandas as pd
import requests
from sqlalchemy import create_engine, text


HOLIDAY_API_TEMPLATE = "https://calendrier.api.gouv.fr/jours-feries/metropole/{year}.json"


def fetch_holidays(year: int, timeout_sec: int = 30) -> dict:
    url = HOLIDAY_API_TEMPLATE.format(year=year)
    resp = requests.get(url, timeout=timeout_sec)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise ValueError("holiday API response must be a dictionary of date -> holiday_name")
    return payload


def holidays_dataframe(
    payload: dict,
    *,
    year: int,
    run_id: str,
    country_code: str = "FR",
    ingested_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    ingested_at = ingested_at or pd.Timestamp.now(tz="UTC")
    rows = []
    for ds, holiday_name in payload.items():
        holiday_date = pd.to_datetime(ds, errors="coerce").date()
        if holiday_date is None or holiday_date.year != year:
            continue
        rows.append(
            {
                "run_id": run_id,
                "ingested_at": ingested_at,
                "country_code": country_code,
                "holiday_date": holiday_date,
                "is_holiday": True,
                "holiday_name": str(holiday_name),
            }
        )
    if not rows:
        raise ValueError(f"No holiday rows parsed for year={year}")
    return pd.DataFrame(rows)


def ensure_stg_holidays(conn_uri: str) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS stg_holidays (
      run_id        TEXT NOT NULL,
      ingested_at   TIMESTAMPTZ NOT NULL,
      country_code  TEXT NOT NULL DEFAULT 'FR',
      holiday_date  DATE NOT NULL,
      is_holiday    BOOLEAN NOT NULL,
      holiday_name  TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_stg_holidays_date
      ON stg_holidays (holiday_date);
    """
    engine = create_engine(conn_uri, pool_pre_ping=True)
    try:
        with engine.begin() as conn:
            conn.execute(text(ddl))
    finally:
        engine.dispose()


def load_holidays_to_staging(conn_uri: str, *, df: pd.DataFrame, year: int, country_code: str = "FR") -> int:
    year_start = date(year, 1, 1)
    year_end = date(year, 12, 31)
    engine = create_engine(conn_uri, pool_pre_ping=True)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    DELETE FROM stg_holidays
                    WHERE country_code = :country_code
                      AND holiday_date BETWEEN :year_start AND :year_end
                    """
                ),
                {"country_code": country_code, "year_start": year_start, "year_end": year_end},
            )
        df.to_sql("stg_holidays", con=engine, if_exists="append", index=False, method="multi", chunksize=1000)
    finally:
        engine.dispose()
    return len(df)


def ensure_dim_date_columns(conn_uri: str) -> None:
    stmts = [
        "ALTER TABLE dim_date ADD COLUMN IF NOT EXISTS is_weekend BOOLEAN;",
        "ALTER TABLE dim_date ADD COLUMN IF NOT EXISTS is_holiday BOOLEAN;",
        "ALTER TABLE dim_date ADD COLUMN IF NOT EXISTS holiday_name TEXT;",
    ]
    engine = create_engine(conn_uri, pool_pre_ping=True)
    try:
        with engine.begin() as conn:
            for stmt in stmts:
                conn.execute(text(stmt))
    finally:
        engine.dispose()


def upsert_dim_date_for_year(conn_uri: str, *, year: int, country_code: str = "FR") -> None:
    year_start = date(year, 1, 1)
    year_end = date(year, 12, 31)
    engine = create_engine(conn_uri, pool_pre_ping=True)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO dim_date (date_id, date, day_of_week, month, year, is_weekend, is_holiday, holiday_name)
                    SELECT
                      TO_CHAR(d::date, 'YYYYMMDD')::int AS date_id,
                      d::date AS date,
                      EXTRACT(ISODOW FROM d)::smallint AS day_of_week,
                      EXTRACT(MONTH FROM d)::smallint AS month,
                      EXTRACT(YEAR FROM d)::smallint AS year,
                      (EXTRACT(ISODOW FROM d) IN (6, 7)) AS is_weekend,
                      FALSE AS is_holiday,
                      NULL::text AS holiday_name
                    FROM generate_series(:year_start::date, :year_end::date, interval '1 day') d
                    ON CONFLICT (date) DO UPDATE
                    SET
                      day_of_week = EXCLUDED.day_of_week,
                      month = EXCLUDED.month,
                      year = EXCLUDED.year,
                      is_weekend = EXCLUDED.is_weekend
                    """
                ),
                {"year_start": year_start.isoformat(), "year_end": year_end.isoformat()},
            )

            conn.execute(
                text(
                    """
                    UPDATE dim_date
                    SET is_holiday = FALSE,
                        holiday_name = NULL
                    WHERE date BETWEEN :year_start AND :year_end
                    """
                ),
                {"year_start": year_start, "year_end": year_end},
            )

            conn.execute(
                text(
                    """
                    UPDATE dim_date dd
                    SET
                      is_holiday = sh.is_holiday,
                      holiday_name = sh.holiday_name
                    FROM stg_holidays sh
                    WHERE dd.date = sh.holiday_date
                      AND sh.country_code = :country_code
                      AND sh.holiday_date BETWEEN :year_start AND :year_end
                    """
                ),
                {"country_code": country_code, "year_start": year_start, "year_end": year_end},
            )
    finally:
        engine.dispose()


def ingest_holidays_year(
    conn_uri: str,
    *,
    year: int,
    run_id: str,
    country_code: str = "FR",
    timeout_sec: int = 30,
) -> dict:
    ensure_stg_holidays(conn_uri)
    ensure_dim_date_columns(conn_uri)
    payload = fetch_holidays(year=year, timeout_sec=timeout_sec)
    df = holidays_dataframe(payload, year=year, run_id=run_id, country_code=country_code)
    rows_written = load_holidays_to_staging(conn_uri, df=df, year=year, country_code=country_code)
    upsert_dim_date_for_year(conn_uri, year=year, country_code=country_code)
    return {
        "year": year,
        "country_code": country_code,
        "rows_written": rows_written,
        "source_url": HOLIDAY_API_TEMPLATE.format(year=year),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest French holidays and update dim_date")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--conn-uri", default=os.getenv("DW_CONN_URI", ""))
    parser.add_argument("--country-code", default=os.getenv("HOLIDAY_COUNTRY_CODE", "FR"))
    parser.add_argument("--timeout-sec", type=int, default=int(os.getenv("HOLIDAY_HTTP_TIMEOUT_SEC", "30")))
    parser.add_argument("--run-id", default=f"manual_holidays_{int(time.time())}")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if not args.conn_uri:
        raise RuntimeError("--conn-uri (or env DW_CONN_URI) is required")
    result = ingest_holidays_year(
        conn_uri=args.conn_uri,
        year=args.year,
        run_id=args.run_id,
        country_code=args.country_code,
        timeout_sec=args.timeout_sec,
    )
    print(json.dumps(result))
