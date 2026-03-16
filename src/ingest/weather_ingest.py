import argparse
import gzip
import io
import json
import os
import time
from datetime import datetime, timezone
from functools import lru_cache

import boto3
import pandas as pd
import requests
from sqlalchemy import create_engine, text


BUCKET = os.getenv("BUCKET")
CITY = os.getenv("CITY", "paris")
API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()
OPENWEATHER_BASE_URL = os.getenv("OPENWEATHER_ONECALL_URL", "https://api.openweathermap.org/data/3.0/onecall")

CITY_COORDS = {
    "nyc": (40.7128, -74.0060),
    "paris": (48.8566, 2.3522),
}


@lru_cache(maxsize=1)
def _default_s3_client():
    return boto3.client("s3")


def floor_to_10min(ts: datetime) -> datetime:
    ts = ts.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return ts.replace(minute=(ts.minute // 10) * 10)


def _coords_for_city(city: str) -> tuple[float, float]:
    coords = CITY_COORDS.get(city.lower().strip())
    if not coords:
        raise ValueError(f"Unsupported city={city!r}. Configure CITY_COORDS in src/ingest/weather_ingest.py")
    return coords


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


def _to_utc(ts_value: int | float | None) -> datetime | None:
    if ts_value is None:
        return None
    return datetime.fromtimestamp(ts_value, tz=timezone.utc)


def _weather_summary(block: dict) -> dict:
    weather_items = block.get("weather") or []
    first = weather_items[0] if weather_items else {}
    return {
        "weather_code": first.get("id"),
        "weather_main": first.get("main"),
        "weather_description": first.get("description"),
    }


def _precipitation_mm(block: dict, *, default: float | None = None) -> float | None:
    total = 0.0
    found = False
    if "precipitation" in block and block.get("precipitation") is not None:
        return float(block.get("precipitation"))

    for kind in ("rain", "snow"):
        value = block.get(kind)
        if isinstance(value, dict):
            hourly = value.get("1h")
            if hourly is not None:
                total += float(hourly)
                found = True
    if found:
        return total
    return default


def _validate_openweather_payload(payload: dict) -> None:
    if not isinstance(payload, dict):
        raise ValueError("OpenWeather payload must be a dict")
    if not isinstance(payload.get("current"), dict):
        raise ValueError("OpenWeather payload missing current block")
    if "dt" not in payload["current"]:
        raise ValueError("OpenWeather current block missing dt")
    if not isinstance(payload.get("hourly", []), list):
        raise ValueError("OpenWeather payload hourly must be a list")


def _snapshot_bucket_from_payload(payload: dict) -> datetime:
    observed_at = _to_utc(payload.get("current", {}).get("dt"))
    if observed_at is None:
        raise ValueError("OpenWeather current block missing valid dt")
    return floor_to_10min(observed_at)


def _ingest_metadata(
    payload: dict,
    *,
    ingested_at: pd.Timestamp | None = None,
) -> dict[str, object]:
    ingested_at = ingested_at or pd.Timestamp.now(tz="UTC")
    snapshot_bucket_at = datetime.fromisoformat(
        payload["_meta_ingest"]["snapshot_bucket_at_utc"]
    ).astimezone(timezone.utc)
    source = payload.get("_meta_ingest", {}).get("source", "openweather-onecall-3.0")
    current = payload.get("current", {})
    observed_at = _to_utc(current.get("dt"))
    if observed_at is None:
        raise ValueError("OpenWeather current block missing valid dt")
    return {
        "ingested_at": ingested_at,
        "snapshot_bucket_at": snapshot_bucket_at,
        "source": source,
        "observed_at": observed_at,
    }


def fetch_weather_payload(
    *,
    city: str,
    api_key: str,
    timeout_sec: int = 30,
) -> dict:
    if not api_key:
        raise RuntimeError("OPENWEATHER_API_KEY is required for weather ingestion")

    lat, lon = _coords_for_city(city)
    response = requests.get(
        OPENWEATHER_BASE_URL,
        params={
            "lat": f"{lat:.4f}",
            "lon": f"{lon:.4f}",
            "appid": api_key,
            "units": "metric",
            "exclude": "minutely,daily,alerts",
        },
        timeout=timeout_sec,
    )
    response.raise_for_status()
    payload = response.json()
    _validate_openweather_payload(payload)
    snapshot_bucket_at = _snapshot_bucket_from_payload(payload)
    payload["_meta_ingest"] = {
        "source": "openweather-onecall-3.0",
        "city": city,
        "lat": lat,
        "lon": lon,
        "snapshot_bucket_at_utc": snapshot_bucket_at.isoformat(),
        "fetched_at_utc": int(time.time()),
    }
    return payload


def persist_weather_raw_to_s3(
    payload: dict,
    *,
    bucket: str,
    city: str,
    run_id: str,
    s3_client=None,
) -> dict:
    client = s3_client or _default_s3_client()
    snapshot_bucket_at = datetime.fromisoformat(payload["_meta_ingest"]["snapshot_bucket_at_utc"]).astimezone(timezone.utc)
    dt_prefix = snapshot_bucket_at.strftime("dt=%Y-%m-%d-%H-%M")
    data_key = f"raw/weather/city={city}/{dt_prefix}/data.json.gz"
    manifest_key = f"raw/weather/city={city}/{dt_prefix}/_manifest.json.gz"

    _put_json(payload, data_key, bucket=bucket, s3_client=client)
    _put_json(
        {
            "city": city,
            "run_id": run_id,
            "source": payload.get("_meta_ingest", {}).get("source"),
            "snapshot_bucket_at_utc": payload.get("_meta_ingest", {}).get("snapshot_bucket_at_utc"),
            "ingested_utc": int(time.time()),
        },
        manifest_key,
        bucket=bucket,
        s3_client=client,
    )
    return {"data_key": data_key, "manifest_key": manifest_key, "prefix": dt_prefix}


def weather_current_dataframe(
    payload: dict,
    *,
    city: str,
    run_id: str,
    ingested_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    metadata = _ingest_metadata(payload, ingested_at=ingested_at)
    current = payload.get("current", {})
    current_weather = _weather_summary(current)

    row = {
        "run_id": run_id,
        "ingested_at": metadata["ingested_at"],
        "city": city,
        "snapshot_bucket_at": metadata["snapshot_bucket_at"],
        "observed_at": metadata["observed_at"],
        "temperature_c": current.get("temp"),
        "humidity_pct": current.get("humidity"),
        "wind_speed_ms": current.get("wind_speed"),
        "precipitation_mm": _precipitation_mm(current, default=0.0),
        "weather_code": current_weather["weather_code"],
        "weather_main": current_weather["weather_main"],
        "weather_description": current_weather["weather_description"],
        "source": metadata["source"],
    }
    return pd.DataFrame([row])


def weather_hourly_dataframe(
    payload: dict,
    *,
    city: str,
    run_id: str,
    ingested_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    metadata = _ingest_metadata(payload, ingested_at=ingested_at)
    current = payload.get("current", {})
    observed_at = metadata["observed_at"]
    window_end = observed_at + pd.Timedelta(hours=1)

    rows = []
    for hourly_row in payload.get("hourly", []):
        forecast_at = _to_utc(hourly_row.get("dt"))
        if forecast_at is None:
            continue
        if not (observed_at < forecast_at <= window_end):
            continue
        summary = _weather_summary(hourly_row)
        rows.append(
            {
                "run_id": run_id,
                "ingested_at": metadata["ingested_at"],
                "city": city,
                "snapshot_bucket_at": metadata["snapshot_bucket_at"],
                "observed_at": observed_at,
                "forecast_at": forecast_at,
                "forecast_horizon_min": int((forecast_at - observed_at).total_seconds() // 60),
                "temperature_c": hourly_row.get("temp"),
                "humidity_pct": hourly_row.get("humidity"),
                "wind_speed_ms": hourly_row.get("wind_speed"),
                "precipitation_mm": _precipitation_mm(hourly_row),
                "precipitation_probability_pct": float(hourly_row.get("pop")) * 100 if hourly_row.get("pop") is not None else None,
                "weather_code": summary["weather_code"],
                "weather_main": summary["weather_main"],
                "weather_description": summary["weather_description"],
                "source": metadata["source"],
            }
        )
    return pd.DataFrame(rows)


def ensure_weather_staging_tables(conn_uri: str) -> None:
    current_table_sql = """
    CREATE TABLE IF NOT EXISTS stg_weather_current (
      run_id               TEXT NOT NULL,
      ingested_at          TIMESTAMPTZ NOT NULL,
      city                 TEXT NOT NULL,
      snapshot_bucket_at   TIMESTAMPTZ NOT NULL,
      observed_at          TIMESTAMPTZ NOT NULL,
      temperature_c        DOUBLE PRECISION,
      humidity_pct         DOUBLE PRECISION,
      wind_speed_ms        DOUBLE PRECISION,
      precipitation_mm     DOUBLE PRECISION,
      weather_code         INTEGER,
      weather_main         TEXT,
      weather_description  TEXT,
      source               TEXT
    );
    """
    hourly_table_sql = """
    CREATE TABLE IF NOT EXISTS stg_weather_hourly (
      run_id                          TEXT NOT NULL,
      ingested_at                     TIMESTAMPTZ NOT NULL,
      city                            TEXT NOT NULL,
      snapshot_bucket_at              TIMESTAMPTZ NOT NULL,
      observed_at                     TIMESTAMPTZ NOT NULL,
      forecast_at                     TIMESTAMPTZ NOT NULL,
      forecast_horizon_min            INTEGER,
      temperature_c                   DOUBLE PRECISION,
      humidity_pct                    DOUBLE PRECISION,
      wind_speed_ms                   DOUBLE PRECISION,
      precipitation_mm                DOUBLE PRECISION,
      precipitation_probability_pct   DOUBLE PRECISION,
      weather_code                    INTEGER,
      weather_main                    TEXT,
      weather_description             TEXT,
      source                          TEXT
    );
    """
    index_sqls = [
        "CREATE INDEX IF NOT EXISTS idx_stg_weather_current_city_bucket ON stg_weather_current (city, snapshot_bucket_at);",
        "CREATE INDEX IF NOT EXISTS idx_stg_weather_current_city_obs ON stg_weather_current (city, observed_at);",
        "CREATE INDEX IF NOT EXISTS idx_stg_weather_hourly_city_bucket ON stg_weather_hourly (city, snapshot_bucket_at);",
        "CREATE INDEX IF NOT EXISTS idx_stg_weather_hourly_city_forecast ON stg_weather_hourly (city, forecast_at);",
    ]
    alter_sqls = [
        "ALTER TABLE stg_weather_current ADD COLUMN IF NOT EXISTS run_id TEXT;",
        "ALTER TABLE stg_weather_current ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ;",
        "ALTER TABLE stg_weather_current ADD COLUMN IF NOT EXISTS city TEXT;",
        "ALTER TABLE stg_weather_current ADD COLUMN IF NOT EXISTS snapshot_bucket_at TIMESTAMPTZ;",
        "ALTER TABLE stg_weather_current ADD COLUMN IF NOT EXISTS observed_at TIMESTAMPTZ;",
        "ALTER TABLE stg_weather_current ADD COLUMN IF NOT EXISTS temperature_c DOUBLE PRECISION;",
        "ALTER TABLE stg_weather_current ADD COLUMN IF NOT EXISTS humidity_pct DOUBLE PRECISION;",
        "ALTER TABLE stg_weather_current ADD COLUMN IF NOT EXISTS wind_speed_ms DOUBLE PRECISION;",
        "ALTER TABLE stg_weather_current ADD COLUMN IF NOT EXISTS precipitation_mm DOUBLE PRECISION;",
        "ALTER TABLE stg_weather_current ADD COLUMN IF NOT EXISTS weather_code INTEGER;",
        "ALTER TABLE stg_weather_current ADD COLUMN IF NOT EXISTS weather_main TEXT;",
        "ALTER TABLE stg_weather_current ADD COLUMN IF NOT EXISTS weather_description TEXT;",
        "ALTER TABLE stg_weather_current ADD COLUMN IF NOT EXISTS source TEXT;",
        "ALTER TABLE stg_weather_hourly ADD COLUMN IF NOT EXISTS run_id TEXT;",
        "ALTER TABLE stg_weather_hourly ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ;",
        "ALTER TABLE stg_weather_hourly ADD COLUMN IF NOT EXISTS city TEXT;",
        "ALTER TABLE stg_weather_hourly ADD COLUMN IF NOT EXISTS snapshot_bucket_at TIMESTAMPTZ;",
        "ALTER TABLE stg_weather_hourly ADD COLUMN IF NOT EXISTS observed_at TIMESTAMPTZ;",
        "ALTER TABLE stg_weather_hourly ADD COLUMN IF NOT EXISTS forecast_at TIMESTAMPTZ;",
        "ALTER TABLE stg_weather_hourly ADD COLUMN IF NOT EXISTS forecast_horizon_min INTEGER;",
        "ALTER TABLE stg_weather_hourly ADD COLUMN IF NOT EXISTS temperature_c DOUBLE PRECISION;",
        "ALTER TABLE stg_weather_hourly ADD COLUMN IF NOT EXISTS humidity_pct DOUBLE PRECISION;",
        "ALTER TABLE stg_weather_hourly ADD COLUMN IF NOT EXISTS wind_speed_ms DOUBLE PRECISION;",
        "ALTER TABLE stg_weather_hourly ADD COLUMN IF NOT EXISTS precipitation_mm DOUBLE PRECISION;",
        "ALTER TABLE stg_weather_hourly ADD COLUMN IF NOT EXISTS precipitation_probability_pct DOUBLE PRECISION;",
        "ALTER TABLE stg_weather_hourly ADD COLUMN IF NOT EXISTS weather_code INTEGER;",
        "ALTER TABLE stg_weather_hourly ADD COLUMN IF NOT EXISTS weather_main TEXT;",
        "ALTER TABLE stg_weather_hourly ADD COLUMN IF NOT EXISTS weather_description TEXT;",
        "ALTER TABLE stg_weather_hourly ADD COLUMN IF NOT EXISTS source TEXT;",
    ]

    engine = create_engine(conn_uri, pool_pre_ping=True)
    try:
        with engine.begin() as conn:
            legacy_tables = {
                row[0]
                for row in conn.execute(
                    text(
                        """
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = 'public'
                          AND table_name IN ('stg_weather', 'stg_weather_current', 'stg_weather_hourly')
                        """
                    )
                )
            }
            if "stg_weather" in legacy_tables:
                raise RuntimeError(
                    "Legacy stg_weather table detected. Drop or rename public.stg_weather before running the refactored weather pipeline."
                )
            conn.execute(text(current_table_sql))
            conn.execute(text(hourly_table_sql))
            conn.execute(text("ALTER TABLE stg_weather_current DROP COLUMN IF EXISTS source_last_updated;"))
            conn.execute(text("ALTER TABLE stg_weather_hourly DROP COLUMN IF EXISTS source_last_updated;"))
            for stmt in alter_sqls:
                conn.execute(text(stmt))
            conn.execute(
                text(
                    """
                    UPDATE stg_weather_current
                    SET snapshot_bucket_at = to_timestamp(
                        floor(extract(epoch from observed_at) / 600.0) * 600
                    )
                    WHERE observed_at IS NOT NULL
                      AND (
                          snapshot_bucket_at IS NULL
                          OR snapshot_bucket_at <> to_timestamp(
                              floor(extract(epoch from observed_at) / 600.0) * 600
                          )
                      )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    UPDATE stg_weather_hourly
                    SET snapshot_bucket_at = to_timestamp(
                        floor(extract(epoch from observed_at) / 600.0) * 600
                    )
                    WHERE observed_at IS NOT NULL
                      AND (
                          snapshot_bucket_at IS NULL
                          OR snapshot_bucket_at <> to_timestamp(
                              floor(extract(epoch from observed_at) / 600.0) * 600
                          )
                      )
                    """
                )
            )
            for stmt in index_sqls:
                conn.execute(text(stmt))
    finally:
        engine.dispose()


def ingest_weather_to_staging(
    conn_uri: str,
    *,
    payload: dict,
    city: str,
    run_id: str,
) -> dict:
    current_df = weather_current_dataframe(payload, city=city, run_id=run_id)
    hourly_df = weather_hourly_dataframe(payload, city=city, run_id=run_id)
    snapshot_bucket_at = pd.Timestamp(current_df.iloc[0]["snapshot_bucket_at"]).to_pydatetime()

    engine = create_engine(conn_uri, pool_pre_ping=True)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    DELETE FROM stg_weather_current
                    WHERE city = :city
                      AND snapshot_bucket_at = :snapshot_bucket_at
                    """
                ),
                {"city": city, "snapshot_bucket_at": snapshot_bucket_at},
            )
            conn.execute(
                text(
                    """
                    DELETE FROM stg_weather_hourly
                    WHERE city = :city
                      AND snapshot_bucket_at = :snapshot_bucket_at
                    """
                ),
                {"city": city, "snapshot_bucket_at": snapshot_bucket_at},
            )
        current_df.to_sql("stg_weather_current", con=engine, if_exists="append", index=False, method="multi", chunksize=1000)
        if not hourly_df.empty:
            hourly_df.to_sql("stg_weather_hourly", con=engine, if_exists="append", index=False, method="multi", chunksize=1000)
    finally:
        engine.dispose()

    return {"current_rows_written": len(current_df), "hourly_rows_written": len(hourly_df)}


def ingest_weather_dual_write(
    conn_uri: str,
    *,
    city: str,
    run_id: str,
    raw_bucket: str,
    api_key: str,
    timeout_sec: int = 30,
) -> dict:
    ensure_weather_staging_tables(conn_uri)
    payload = fetch_weather_payload(
        city=city,
        api_key=api_key,
        timeout_sec=timeout_sec,
    )
    raw_result = persist_weather_raw_to_s3(
        payload,
        bucket=raw_bucket,
        city=city,
        run_id=run_id,
    )
    write_result = ingest_weather_to_staging(
        conn_uri,
        payload=payload,
        city=city,
        run_id=run_id,
    )
    return {
        **write_result,
        "raw": raw_result,
        "snapshot_bucket_at_utc": payload.get("_meta_ingest", {}).get("snapshot_bucket_at_utc"),
        "source": payload.get("_meta_ingest", {}).get("source"),
    }


def handler(event, context):
    bucket = BUCKET
    if not bucket:
        raise RuntimeError("Env BUCKET is required")

    city = os.getenv("CITY", CITY)
    api_key = os.getenv("OPENWEATHER_API_KEY", API_KEY)
    payload = fetch_weather_payload(
        city=city,
        api_key=api_key,
        timeout_sec=int(os.getenv("WEATHER_HTTP_TIMEOUT_SEC", "30")),
    )
    raw_result = persist_weather_raw_to_s3(
        payload,
        bucket=bucket,
        city=city,
        run_id=f"handler_{int(time.time())}",
    )
    return {"ok": True, "raw": raw_result, "source": payload.get("_meta_ingest", {}).get("source")}


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
    parser = argparse.ArgumentParser(
        description="Ingest OpenWeather current and hourly forecast data to S3 raw and Postgres staging"
    )
    parser.add_argument("--city", default=os.getenv("CITY", CITY))
    parser.add_argument("--raw-bucket", default=os.getenv("BUCKET", BUCKET or ""))
    parser.add_argument("--conn-uri", default=_build_conn_uri_from_env())
    parser.add_argument("--run-id", default=f"manual_{int(time.time())}")
    parser.add_argument("--api-key", default=os.getenv("OPENWEATHER_API_KEY", API_KEY))
    parser.add_argument("--timeout-sec", type=int, default=int(os.getenv("WEATHER_HTTP_TIMEOUT_SEC", "30")))
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if not args.conn_uri:
        raise RuntimeError("--conn-uri (or env DW_CONN_URI / PGHOST+PGDATABASE+PGUSER+PGPASSWORD) is required")
    if not args.raw_bucket:
        raise RuntimeError("--raw-bucket (or env BUCKET) is required")
    if not args.api_key:
        raise RuntimeError("--api-key (or env OPENWEATHER_API_KEY) is required")

    result = ingest_weather_dual_write(
        conn_uri=args.conn_uri,
        city=args.city,
        run_id=args.run_id,
        raw_bucket=args.raw_bucket,
        api_key=args.api_key,
        timeout_sec=args.timeout_sec,
    )
    print(json.dumps(result))
