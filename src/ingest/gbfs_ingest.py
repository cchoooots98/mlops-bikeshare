import argparse
import gzip
import io
import json
import os
import time
from datetime import datetime, timezone
from functools import lru_cache

import boto3
import botocore
import pandas as pd
import requests
from sqlalchemy import create_engine, text

from .validators import validate_station_info, validate_station_status

BUCKET = os.getenv("BUCKET")
CITY = os.getenv("CITY", "paris")

GBFS_ROOT = {
    "nyc": "https://gbfs.citibikenyc.com/gbfs/en/gbfs.json",
    "paris": "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/gbfs.json",
}


@lru_cache(maxsize=1)
def _default_s3_client():
    return boto3.client("s3")


def floor_to_5min(ts: datetime) -> datetime:
    ts = ts.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return ts.replace(minute=(ts.minute // 5) * 5)


def _dt_prefix_from_epoch(epoch_sec: int) -> str:
    dt = datetime.fromtimestamp(epoch_sec, tz=timezone.utc)
    dt5 = floor_to_5min(dt)
    return dt5.strftime("dt=%Y-%m-%d-%H-%M")


def _snapshot_bucket_from_payload(payload: dict) -> datetime:
    return floor_to_5min(datetime.fromtimestamp(_last_updated_epoch(payload), tz=timezone.utc))


def _exists(key: str, bucket: str, s3_client=None) -> bool:
    client = s3_client or _default_s3_client()
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError:
        return False


def _put_json(obj: dict, key: str, bucket: str, s3_client=None):
    client = s3_client or _default_s3_client()
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    gzbuf = io.BytesIO()
    with gzip.GzipFile(fileobj=gzbuf, mode="wb") as gz:
        gz.write(body)
    client.put_object(
        Bucket=bucket, Key=key, Body=gzbuf.getvalue(), ContentType="application/json", ContentEncoding="gzip"
    )


def _get_json(url: str, timeout_sec: int = 10) -> dict:
    resp = requests.get(url, timeout=timeout_sec)
    resp.raise_for_status()
    return resp.json()


def _find_feed_url(gbfs_payload: dict, feed_name: str) -> str | None:
    data = gbfs_payload.get("data", {})
    for lang_payload in data.values():
        feeds = lang_payload.get("feeds", []) if isinstance(lang_payload, dict) else []
        for feed in feeds:
            if feed.get("name") == feed_name and feed.get("url"):
                return feed["url"]
    return None


def _resolve_feed_urls(gbfs_root_url: str) -> tuple[str, str]:
    gbfs = _get_json(gbfs_root_url)
    status_url = _find_feed_url(gbfs, "station_status")
    info_url = _find_feed_url(gbfs, "station_information")

    if status_url and info_url:
        return info_url, status_url

    base = gbfs_root_url.replace("/gbfs.json", "")
    return (
        info_url or f"{base}/station_information.json",
        status_url or f"{base}/station_status.json",
    )


def _last_updated_epoch(payload: dict) -> int:
    for key in ("last_updated", "lastUpdatedOther"):
        value = payload.get(key)
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    raise ValueError("Missing valid last_updated / lastUpdatedOther in GBFS payload")


def persist_raw_payload_to_s3(
    payload: dict,
    *,
    bucket: str,
    city: str,
    feed_name: str,
    run_id: str | None = None,
    upsert: bool = True,
    s3_client=None,
) -> dict:
    source_ts = _last_updated_epoch(payload)
    snapshot_bucket_at = _snapshot_bucket_from_payload(payload)
    dt_prefix = snapshot_bucket_at.strftime("dt=%Y-%m-%d-%H-%M")
    data_key = f"raw/{feed_name}/city={city}/{dt_prefix}/data.json.gz"
    manifest_key = f"raw/{feed_name}/city={city}/{dt_prefix}/_manifest.json.gz"

    if not upsert and _exists(data_key, bucket=bucket, s3_client=s3_client):
        return {"skipped": True, "data_key": data_key, "manifest_key": manifest_key, "prefix": dt_prefix}

    _put_json(payload, data_key, bucket=bucket, s3_client=s3_client)
    _put_json(
        {
            "city": city,
            "feed_name": feed_name,
            "run_id": run_id,
            "source_ts": source_ts,
            "snapshot_bucket_at_utc": snapshot_bucket_at.isoformat(),
            "ingested_utc": int(time.time()),
        },
        manifest_key,
        bucket=bucket,
        s3_client=s3_client,
    )
    return {"skipped": False, "data_key": data_key, "manifest_key": manifest_key, "prefix": dt_prefix}


def fetch_station_information_payload(gbfs_root_url: str, timeout_sec: int = 30) -> dict:
    info_url, _ = _resolve_feed_urls(gbfs_root_url)
    return _get_json(info_url, timeout_sec=timeout_sec)


def fetch_station_status_payload(gbfs_root_url: str, timeout_sec: int = 30) -> dict:
    _, status_url = _resolve_feed_urls(gbfs_root_url)
    return _get_json(status_url, timeout_sec=timeout_sec)


def fetch_station_payloads(gbfs_root_url: str, timeout_sec: int = 30) -> tuple[dict, dict]:
    info_url, status_url = _resolve_feed_urls(gbfs_root_url)
    info = _get_json(info_url, timeout_sec=timeout_sec)
    status = _get_json(status_url, timeout_sec=timeout_sec)
    return info, status


def _series_from_candidates(df: pd.DataFrame, candidates: list[str], default=None) -> pd.Series:
    result = pd.Series([default] * len(df), dtype="object")
    for col in candidates:
        if col in df.columns:
            result = result.fillna(df[col])
    return result


def station_information_dataframe(
    payload: dict,
    run_id: str,
    city: str,
    ingested_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    stations = payload.get("data", {}).get("stations", [])
    if not stations:
        raise ValueError("station_information has no rows")

    df = pd.DataFrame(stations)
    ingested_at = ingested_at or pd.Timestamp.now(tz="UTC")
    source_last_updated = _last_updated_epoch(payload)
    snapshot_bucket_at = _snapshot_bucket_from_payload(payload)

    return pd.DataFrame(
        {
            "run_id": run_id,
            "ingested_at": ingested_at,
            "source_last_updated": source_last_updated,
            "city": city,
            "snapshot_bucket_at": snapshot_bucket_at,
            "station_id": _series_from_candidates(df, ["station_id"]).astype(str),
            "name": _series_from_candidates(df, ["name"]),
            "lat": pd.to_numeric(_series_from_candidates(df, ["lat"]), errors="coerce"),
            "lon": pd.to_numeric(_series_from_candidates(df, ["lon"]), errors="coerce"),
            "capacity": pd.to_numeric(_series_from_candidates(df, ["capacity"]), errors="coerce").astype("Int64"),
        }
    )


def station_status_dataframe(
    payload: dict,
    run_id: str,
    city: str,
    ingested_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    stations = payload.get("data", {}).get("stations", [])
    if not stations:
        raise ValueError("station_status has no rows")

    df = pd.DataFrame(stations)
    ingested_at = ingested_at or pd.Timestamp.now(tz="UTC")
    source_last_updated = _last_updated_epoch(payload)
    snapshot_bucket_at = _snapshot_bucket_from_payload(payload)

    bikes = pd.to_numeric(
        _series_from_candidates(df, ["num_bikes_available", "numBikesAvailable"]),
        errors="coerce",
    ).astype("Int64")
    docks = pd.to_numeric(
        _series_from_candidates(df, ["num_docks_available", "numDocksAvailable"]),
        errors="coerce",
    ).astype("Int64")
    renting = pd.to_numeric(_series_from_candidates(df, ["is_renting"]), errors="coerce").astype("Int64")
    returning = pd.to_numeric(_series_from_candidates(df, ["is_returning"]), errors="coerce").astype("Int64")
    last_reported_epoch = pd.to_numeric(_series_from_candidates(df, ["last_reported"]), errors="coerce")

    return pd.DataFrame(
        {
            "run_id": run_id,
            "ingested_at": ingested_at,
            "source_last_updated": source_last_updated,
            "city": city,
            "snapshot_bucket_at": snapshot_bucket_at,
            "station_id": _series_from_candidates(df, ["station_id"]).astype(str),
            "num_bikes_available": bikes,
            "num_docks_available": docks,
            "is_renting": renting,
            "is_returning": returning,
            "last_reported_at": pd.to_datetime(last_reported_epoch, unit="s", utc=True, errors="coerce"),
        }
    )


def ensure_staging_tables(conn_uri: str) -> None:
    create_info_sql = """
    CREATE TABLE IF NOT EXISTS stg_station_information (
        run_id TEXT NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        source_last_updated BIGINT,
        city TEXT NOT NULL,
        snapshot_bucket_at TIMESTAMPTZ NOT NULL,
        station_id TEXT NOT NULL,
        name TEXT,
        lat DOUBLE PRECISION,
        lon DOUBLE PRECISION,
        capacity INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_stg_station_information_ingested_at
        ON stg_station_information (ingested_at);
    """
    create_status_sql = """
    CREATE TABLE IF NOT EXISTS stg_station_status (
        run_id TEXT NOT NULL,
        ingested_at TIMESTAMPTZ NOT NULL,
        source_last_updated BIGINT,
        city TEXT NOT NULL,
        snapshot_bucket_at TIMESTAMPTZ NOT NULL,
        station_id TEXT NOT NULL,
        num_bikes_available INTEGER,
        num_docks_available INTEGER,
        is_renting SMALLINT,
        is_returning SMALLINT,
        last_reported_at TIMESTAMPTZ
    );
    CREATE INDEX IF NOT EXISTS idx_stg_station_status_ingested_at
        ON stg_station_status (ingested_at);
    """

    engine = create_engine(conn_uri, pool_pre_ping=True)
    try:
        with engine.begin() as conn:
            conn.execute(text(create_info_sql))
            conn.execute(text(create_status_sql))
            conn.execute(text("ALTER TABLE stg_station_information ADD COLUMN IF NOT EXISTS city TEXT;"))
            conn.execute(text("ALTER TABLE stg_station_status ADD COLUMN IF NOT EXISTS city TEXT;"))
            conn.execute(
                text("ALTER TABLE stg_station_information ADD COLUMN IF NOT EXISTS snapshot_bucket_at TIMESTAMPTZ;")
            )
            conn.execute(
                text("ALTER TABLE stg_station_status ADD COLUMN IF NOT EXISTS snapshot_bucket_at TIMESTAMPTZ;")
            )
            conn.execute(text("UPDATE stg_station_information SET city = :city WHERE city IS NULL;"), {"city": CITY})
            conn.execute(text("UPDATE stg_station_status SET city = :city WHERE city IS NULL;"), {"city": CITY})
            conn.execute(
                text(
                    """
                    UPDATE stg_station_information
                    SET snapshot_bucket_at = coalesce(
                        to_timestamp(floor(source_last_updated / 300.0) * 300),
                        to_timestamp(floor(extract(epoch from ingested_at) / 300.0) * 300)
                    )
                    WHERE snapshot_bucket_at IS NULL
                    """
                )
            )
            conn.execute(
                text(
                    """
                    UPDATE stg_station_status
                    SET snapshot_bucket_at = coalesce(
                        to_timestamp(floor(source_last_updated / 300.0) * 300),
                        to_timestamp(floor(extract(epoch from ingested_at) / 300.0) * 300)
                    )
                    WHERE snapshot_bucket_at IS NULL
                    """
                )
            )
            conn.execute(text("ALTER TABLE stg_station_information ALTER COLUMN city SET NOT NULL;"))
            conn.execute(text("ALTER TABLE stg_station_status ALTER COLUMN city SET NOT NULL;"))
            conn.execute(text("ALTER TABLE stg_station_information ALTER COLUMN snapshot_bucket_at SET NOT NULL;"))
            conn.execute(text("ALTER TABLE stg_station_status ALTER COLUMN snapshot_bucket_at SET NOT NULL;"))
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_stg_station_information_city_station "
                    "ON stg_station_information (city, station_id);"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_stg_station_status_city_station "
                    "ON stg_station_status (city, station_id);"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_stg_station_information_city_bucket "
                    "ON stg_station_information (city, snapshot_bucket_at);"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_stg_station_status_city_bucket "
                    "ON stg_station_status (city, snapshot_bucket_at);"
                )
            )
    finally:
        engine.dispose()


def ingest_station_information_to_staging(
    conn_uri: str,
    gbfs_root_url: str,
    run_id: str,
    timeout_sec: int = 30,
    raw_bucket: str | None = None,
    raw_city: str = "paris",
    raw_upsert: bool = True,
    s3_client=None,
) -> dict:
    payload = fetch_station_information_payload(gbfs_root_url, timeout_sec=timeout_sec)
    validate_station_info(payload)
    raw_result = None
    if raw_bucket:
        raw_result = persist_raw_payload_to_s3(
            payload,
            bucket=raw_bucket,
            city=raw_city,
            feed_name="station_information",
            run_id=run_id,
            upsert=raw_upsert,
            s3_client=s3_client,
        )
    out = station_information_dataframe(payload, run_id=run_id, city=raw_city)
    snapshot_bucket_at = pd.Timestamp(out.iloc[0]["snapshot_bucket_at"]).to_pydatetime()

    engine = create_engine(conn_uri, pool_pre_ping=True)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    DELETE FROM stg_station_information
                    WHERE city = :city
                      AND snapshot_bucket_at = :snapshot_bucket_at
                    """
                ),
                {"city": raw_city, "snapshot_bucket_at": snapshot_bucket_at},
            )
        out.to_sql(
            "stg_station_information", con=engine, if_exists="append", index=False, method="multi", chunksize=1000
        )
    finally:
        engine.dispose()
    return {"rows_written": len(out), "raw": raw_result, "snapshot_bucket_at_utc": snapshot_bucket_at.isoformat()}


def ingest_station_status_to_staging(
    conn_uri: str,
    gbfs_root_url: str,
    run_id: str,
    timeout_sec: int = 30,
    raw_bucket: str | None = None,
    raw_city: str = "paris",
    raw_upsert: bool = True,
    s3_client=None,
) -> dict:
    payload = fetch_station_status_payload(gbfs_root_url, timeout_sec=timeout_sec)
    validate_station_status(payload)
    raw_result = None
    if raw_bucket:
        raw_result = persist_raw_payload_to_s3(
            payload,
            bucket=raw_bucket,
            city=raw_city,
            feed_name="station_status",
            run_id=run_id,
            upsert=raw_upsert,
            s3_client=s3_client,
        )
    out = station_status_dataframe(payload, run_id=run_id, city=raw_city)
    snapshot_bucket_at = pd.Timestamp(out.iloc[0]["snapshot_bucket_at"]).to_pydatetime()

    engine = create_engine(conn_uri, pool_pre_ping=True)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    DELETE FROM stg_station_status
                    WHERE city = :city
                      AND snapshot_bucket_at = :snapshot_bucket_at
                    """
                ),
                {"city": raw_city, "snapshot_bucket_at": snapshot_bucket_at},
            )
        out.to_sql("stg_station_status", con=engine, if_exists="append", index=False, method="multi", chunksize=1000)
    finally:
        engine.dispose()
    return {"rows_written": len(out), "raw": raw_result, "snapshot_bucket_at_utc": snapshot_bucket_at.isoformat()}


def handler(event, context):
    if not BUCKET:
        raise RuntimeError("Env BUCKET is required")

    root = GBFS_ROOT[CITY]
    info, status = fetch_station_payloads(root)

    validate_station_status(status)
    validate_station_info(info)

    info_raw = persist_raw_payload_to_s3(
        info,
        bucket=BUCKET,
        city=CITY,
        feed_name="station_information",
        run_id=None,
        upsert=False,
    )
    status_raw = persist_raw_payload_to_s3(
        status,
        bucket=BUCKET,
        city=CITY,
        feed_name="station_status",
        run_id=None,
        upsert=False,
    )
    return {"ok": True, "info_raw": info_raw, "status_raw": status_raw}


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


def ingest_gbfs_to_staging(
    *,
    conn_uri: str,
    city: str,
    run_id: str,
    gbfs_root_url: str,
    timeout_sec: int = 30,
    raw_bucket: str | None = None,
) -> dict:
    ensure_staging_tables(conn_uri=conn_uri)
    info_result = ingest_station_information_to_staging(
        conn_uri=conn_uri,
        gbfs_root_url=gbfs_root_url,
        run_id=run_id,
        timeout_sec=timeout_sec,
        raw_bucket=raw_bucket,
        raw_city=city,
    )
    status_result = ingest_station_status_to_staging(
        conn_uri=conn_uri,
        gbfs_root_url=gbfs_root_url,
        run_id=run_id,
        timeout_sec=timeout_sec,
        raw_bucket=raw_bucket,
        raw_city=city,
    )
    return {
        "ok": True,
        "city": city,
        "run_id": run_id,
        "raw": raw_bucket is not None,
        "station_information": info_result,
        "station_status": status_result,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest GBFS station information/status into Postgres staging, optionally dual-writing raw S3."
    )
    parser.add_argument("--city", default=CITY)
    parser.add_argument("--gbfs-root-url", default="")
    parser.add_argument("--conn-uri", default=_build_conn_uri_from_env())
    parser.add_argument("--run-id", default=f"manual_gbfs_{int(time.time())}")
    parser.add_argument("--timeout-sec", type=int, default=int(os.getenv("GBFS_HTTP_TIMEOUT_SEC", "30")))
    parser.add_argument("--raw-bucket", default=os.getenv("BUCKET", BUCKET or ""))
    parser.add_argument(
        "--feed",
        choices=["both", "station_information", "station_status"],
        default="both",
        help="Choose which GBFS feed to ingest after staging tables are ensured.",
    )
    parser.add_argument(
        "--ensure-only",
        action="store_true",
        help="Only create/repair staging tables and exit without fetching any feeds.",
    )
    parser.add_argument(
        "--staging-only",
        action="store_true",
        help="Skip raw S3 backup and only write Postgres staging tables.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if not args.conn_uri:
        raise RuntimeError("--conn-uri (or env DW_CONN_URI / PGHOST+PGDATABASE+PGUSER+PGPASSWORD) is required")

    city = args.city.strip().lower()
    if city not in GBFS_ROOT and not args.gbfs_root_url:
        raise RuntimeError(f"Unsupported city '{args.city}'. Provide --gbfs-root-url for a custom feed.")

    gbfs_root_url = args.gbfs_root_url.strip() or GBFS_ROOT[city]
    ensure_staging_tables(conn_uri=args.conn_uri)

    if args.ensure_only:
        result = {"ok": True, "city": city, "ensured_staging_tables": True}
        print(json.dumps(result))
        raise SystemExit(0)

    raw_bucket = None if args.staging_only else (args.raw_bucket.strip() or None)
    if args.feed == "station_information":
        result = {
            "ok": True,
            "city": city,
            "run_id": args.run_id,
            "raw": raw_bucket is not None,
            "station_information": ingest_station_information_to_staging(
                conn_uri=args.conn_uri,
                gbfs_root_url=gbfs_root_url,
                run_id=args.run_id,
                timeout_sec=args.timeout_sec,
                raw_bucket=raw_bucket,
                raw_city=city,
            ),
        }
    elif args.feed == "station_status":
        result = {
            "ok": True,
            "city": city,
            "run_id": args.run_id,
            "raw": raw_bucket is not None,
            "station_status": ingest_station_status_to_staging(
                conn_uri=args.conn_uri,
                gbfs_root_url=gbfs_root_url,
                run_id=args.run_id,
                timeout_sec=args.timeout_sec,
                raw_bucket=raw_bucket,
                raw_city=city,
            ),
        }
    else:
        result = ingest_gbfs_to_staging(
            conn_uri=args.conn_uri,
            city=city,
            run_id=args.run_id,
            gbfs_root_url=gbfs_root_url,
            timeout_sec=args.timeout_sec,
            raw_bucket=raw_bucket,
        )
    print(json.dumps(result))
