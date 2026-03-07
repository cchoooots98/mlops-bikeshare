import argparse
import gzip
import io
import json
import os
import time
from datetime import date, datetime, timezone

import boto3
import pandas as pd
import requests
from sqlalchemy import create_engine, text


HOLIDAY_API_TEMPLATE = "https://calendrier.api.gouv.fr/jours-feries/metropole/{year}.json"
BUCKET = os.getenv("BUCKET")
s3 = boto3.client("s3")


def fetch_holidays(year: int, timeout_sec: int = 30) -> dict:
    url = HOLIDAY_API_TEMPLATE.format(year=year)
    resp = requests.get(url, timeout=timeout_sec)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise ValueError("holiday API response must be a dictionary of date -> holiday_name")
    return payload


def _put_json(obj: dict, key: str, *, bucket: str | None = None, s3_client=None) -> None:
    client = s3_client or s3
    bucket_name = bucket or BUCKET
    if not bucket_name:
        raise RuntimeError("S3 bucket is required")

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
    client.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=buf.getvalue(),
        ContentType="application/json",
        ContentEncoding="gzip",
    )


def persist_holidays_raw_to_s3(
    payload: dict,
    *,
    bucket: str,
    year: int,
    country_code: str,
    run_id: str,
    fetched_at_utc: datetime | None = None,
    s3_client=None,
) -> dict:
    client = s3_client or s3
    fetched_at_utc = (fetched_at_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    dt_prefix = fetched_at_utc.strftime("dt=%Y-%m-%d-%H-%M")
    data_key = f"raw/holidays/country={country_code}/year={year}/{dt_prefix}/data.json.gz"
    manifest_key = f"raw/holidays/country={country_code}/year={year}/{dt_prefix}/_manifest.json.gz"

    _put_json(payload, data_key, bucket=bucket, s3_client=client)
    _put_json(
        {
            "country_code": country_code,
            "year": year,
            "run_id": run_id,
            "source_url": HOLIDAY_API_TEMPLATE.format(year=year),
            "fetched_at_utc": fetched_at_utc.isoformat(),
        },
        manifest_key,
        bucket=bucket,
        s3_client=client,
    )
    return {"data_key": data_key, "manifest_key": manifest_key, "prefix": dt_prefix}


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


def ensure_dim_date_table(conn_uri: str) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS dim_date (
      date_id         INTEGER PRIMARY KEY,
      date            DATE UNIQUE,
      day_of_week     SMALLINT,
      month           SMALLINT,
      year            SMALLINT,
      is_weekend      BOOLEAN,
      is_holiday      BOOLEAN,
      holiday_name    TEXT
    );
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
                    FROM generate_series(:year_start, :year_end, interval '1 day') d
                    ON CONFLICT (date) DO UPDATE
                    SET
                      day_of_week = EXCLUDED.day_of_week,
                      month = EXCLUDED.month,
                      year = EXCLUDED.year,
                      is_weekend = EXCLUDED.is_weekend
                    """
                ),
                {"year_start": year_start, "year_end": year_end},
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
    raw_bucket: str | None = None,
) -> dict:
    ensure_stg_holidays(conn_uri)
    ensure_dim_date_table(conn_uri)
    ensure_dim_date_columns(conn_uri)
    fetched_at_utc = datetime.now(timezone.utc)
    payload = fetch_holidays(year=year, timeout_sec=timeout_sec)
    raw_result = None
    if raw_bucket:
        raw_result = persist_holidays_raw_to_s3(
            payload,
            bucket=raw_bucket,
            year=year,
            country_code=country_code,
            run_id=run_id,
            fetched_at_utc=fetched_at_utc,
        )
    df = holidays_dataframe(payload, year=year, run_id=run_id, country_code=country_code)
    rows_written = load_holidays_to_staging(conn_uri, df=df, year=year, country_code=country_code)
    upsert_dim_date_for_year(conn_uri, year=year, country_code=country_code)
    result = {
        "year": year,
        "country_code": country_code,
        "rows_written": rows_written,
        "source_url": HOLIDAY_API_TEMPLATE.format(year=year),
    }
    if raw_result:
        result["raw"] = raw_result
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest French holidays and update dim_date")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--conn-uri", default=os.getenv("DW_CONN_URI", ""))
    parser.add_argument("--country-code", default=os.getenv("HOLIDAY_COUNTRY_CODE", "FR"))
    parser.add_argument("--timeout-sec", type=int, default=int(os.getenv("HOLIDAY_HTTP_TIMEOUT_SEC", "30")))
    parser.add_argument("--run-id", default=f"manual_holidays_{int(time.time())}")
    parser.add_argument("--raw-bucket", default=os.getenv("RAW_S3_BUCKET", BUCKET or ""))
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
        raw_bucket=args.raw_bucket or None,
    )
    print(json.dumps(result))
