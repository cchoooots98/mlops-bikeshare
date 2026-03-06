import gzip
import io
import json
import os
import time
from datetime import datetime, timezone

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

s3 = boto3.client("s3")


def floor_to_5min(ts: datetime) -> datetime:
    ts = ts.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return ts.replace(minute=(ts.minute // 5) * 5)



def _dt_prefix_from_epoch(epoch_sec: int) -> str:
    dt = datetime.fromtimestamp(epoch_sec, tz=timezone.utc)
    dt5 = floor_to_5min(dt)
    return dt5.strftime("dt=%Y-%m-%d-%H-%M")


def _exists(key: str, bucket: str, s3_client=None) -> bool:
    client = s3_client or s3
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError:
        return False


def _put_json(obj: dict, key: str, bucket: str, s3_client=None):
    client = s3_client or s3
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    gzbuf = io.BytesIO()
    with gzip.GzipFile(fileobj=gzbuf, mode="wb") as gz:
        gz.write(body)
    client.put_object(Bucket=bucket, Key=key, Body=gzbuf.getvalue(), ContentType="application/json", ContentEncoding="gzip")


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
    dt_prefix = _dt_prefix_from_epoch(source_ts)
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


def station_information_dataframe(payload: dict, run_id: str, ingested_at: pd.Timestamp | None = None) -> pd.DataFrame:
    stations = payload.get("data", {}).get("stations", [])
    if not stations:
        raise ValueError("station_information has no rows")

    df = pd.DataFrame(stations)
    ingested_at = ingested_at or pd.Timestamp.now(tz="UTC")

    return pd.DataFrame(
        {
            "run_id": run_id,
            "ingested_at": ingested_at,
            "source_last_updated": _last_updated_epoch(payload),
            "station_id": _series_from_candidates(df, ["station_id"]).astype(str),
            "name": _series_from_candidates(df, ["name"]),
            "lat": pd.to_numeric(_series_from_candidates(df, ["lat"]), errors="coerce"),
            "lon": pd.to_numeric(_series_from_candidates(df, ["lon"]), errors="coerce"),
            "capacity": pd.to_numeric(_series_from_candidates(df, ["capacity"]), errors="coerce").astype("Int64"),
        }
    )


def station_status_dataframe(payload: dict, run_id: str, ingested_at: pd.Timestamp | None = None) -> pd.DataFrame:
    stations = payload.get("data", {}).get("stations", [])
    if not stations:
        raise ValueError("station_status has no rows")

    df = pd.DataFrame(stations)
    ingested_at = ingested_at or pd.Timestamp.now(tz="UTC")

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
            "source_last_updated": _last_updated_epoch(payload),
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
    out = station_information_dataframe(payload, run_id=run_id)

    engine = create_engine(conn_uri, pool_pre_ping=True)
    try:
        out.to_sql("stg_station_information", con=engine, if_exists="append", index=False, method="multi", chunksize=1000)
    finally:
        engine.dispose()
    return {"rows_written": len(out), "raw": raw_result}


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
    out = station_status_dataframe(payload, run_id=run_id)

    engine = create_engine(conn_uri, pool_pre_ping=True)
    try:
        out.to_sql("stg_station_status", con=engine, if_exists="append", index=False, method="multi", chunksize=1000)
    finally:
        engine.dispose()
    return {"rows_written": len(out), "raw": raw_result}


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


if __name__ == "__main__":
    BUCKET = os.getenv("BUCKET")
    CITY = os.getenv("CITY", "paris")
    if not BUCKET:
        raise RuntimeError("Env BUCKET is required, e.g. BUCKET=mlops-bikeshare-...")

    print(f"[gbfs_ingest] city={CITY}, bucket=s3://{BUCKET}")
    result = handler({}, None)
    print("[gbfs_ingest] result:", json.dumps(result))
