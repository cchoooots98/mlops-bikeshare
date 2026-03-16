import argparse
import gzip
import io
import json
import os
import time
from datetime import date, datetime, timezone
from functools import lru_cache

import boto3
import pandas as pd
import requests
from sqlalchemy import create_engine, text


HOLIDAY_API_TEMPLATE = "https://calendrier.api.gouv.fr/jours-feries/metropole/{year}.json"
BUCKET = os.getenv("BUCKET")


@lru_cache(maxsize=1)
def _default_s3_client():
    return boto3.client("s3")


def fetch_holidays(year: int, timeout_sec: int = 30) -> dict:
    url = HOLIDAY_API_TEMPLATE.format(year=year)
    resp = requests.get(url, timeout=timeout_sec)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise ValueError("holiday API response must be a dictionary of date -> holiday_name")
    return payload


def _put_json(obj: dict, key: str, *, bucket: str | None = None, s3_client=None) -> None:
    client = s3_client or _default_s3_client()
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
    client = s3_client or _default_s3_client()
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
    result = {
        "year": year,
        "country_code": country_code,
        "rows_written": rows_written,
        "source_url": HOLIDAY_API_TEMPLATE.format(year=year),
    }
    if raw_result:
        result["raw"] = raw_result
    return result


def _build_conn_uri_from_env() -> str:
    direct_uri = os.getenv("DW_CONN_URI", "").strip()
    if direct_uri:
        return direct_uri

    host = os.getenv("PGHOST", "").strip()
    port = os.getenv("PGPORT", "5432").strip()
    database = os.getenv("PGDATABASE", "").strip()
    user = os.getenv("PGUSER", "").strip()
    password = os.getenv("PGPASSWORD", "").strip()
    if host and database and user and password:
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    return ""


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest French holidays to raw S3 and stg_holidays")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--conn-uri", default=_build_conn_uri_from_env())
    parser.add_argument("--country-code", default=os.getenv("HOLIDAY_COUNTRY_CODE", "FR"))
    parser.add_argument("--timeout-sec", type=int, default=int(os.getenv("HOLIDAY_HTTP_TIMEOUT_SEC", "30")))
    parser.add_argument("--run-id", default=f"manual_holidays_{int(time.time())}")
    parser.add_argument("--raw-bucket", default=os.getenv("BUCKET", BUCKET or ""))
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if not args.conn_uri:
        raise RuntimeError("--conn-uri (or env DW_CONN_URI / PGHOST+PGDATABASE+PGUSER+PGPASSWORD) is required")
    if not args.raw_bucket:
        raise RuntimeError("--raw-bucket (or env BUCKET) is required")
    result = ingest_holidays_year(
        conn_uri=args.conn_uri,
        year=args.year,
        run_id=args.run_id,
        country_code=args.country_code,
        timeout_sec=args.timeout_sec,
        raw_bucket=args.raw_bucket,
    )
    print(json.dumps(result))
