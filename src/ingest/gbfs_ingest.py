import gzip
import io
import json
import os
import time
from datetime import datetime, timezone

import boto3
import botocore
import requests

from .validators import validate_station_info, validate_station_status


def floor_to_5min(ts: datetime) -> datetime:
    ts = ts.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return ts.replace(minute=(ts.minute // 5) * 5)


BUCKET = os.getenv("BUCKET")
CITY = os.getenv("CITY", "paris")

GBFS_ROOT = {
    # Official Vélib' GBFS entrypoint (feed discovery JSON)
    "paris": "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/gbfs.json",
}

s3 = boto3.client("s3")


# def _dt_prefix(epoch_sec:int)->str:
#     # 以源 last_updated 对齐分钟；统一 UTC
#     dt = datetime.fromtimestamp(epoch_sec, tz=timezone.utc)
#     return dt.strftime("dt=%Y-%m-%d-%H-%M")
def _dt_prefix_from_epoch(epoch_sec: int) -> str:
    # 以源 last_updated 为准并向下取整到 5 分钟，统一 UTC
    dt = datetime.fromtimestamp(epoch_sec, tz=timezone.utc)
    dt5 = floor_to_5min(dt)
    return dt5.strftime("dt=%Y-%m-%d-%H-%M")


def _exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except botocore.exceptions.ClientError:
        return False


def _put_json(obj: dict, key: str):
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    # 开启 gzip（省流量&快速下载）
    gzbuf = io.BytesIO()
    with gzip.GzipFile(fileobj=gzbuf, mode="wb") as gz:
        gz.write(body)
    s3.put_object(Bucket=BUCKET, Key=key, Body=gzbuf.getvalue(), ContentType="application/json", ContentEncoding="gzip")


def _get_json(url: str) -> dict:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _find_feed_url(gbfs_payload: dict, feed_name: str) -> str | None:
    # GBFS spec usually stores feeds under data.<lang>.feeds
    data = gbfs_payload.get("data", {})
    for lang_payload in data.values():
        feeds = lang_payload.get("feeds", []) if isinstance(lang_payload, dict) else []
        for f in feeds:
            if f.get("name") == feed_name and f.get("url"):
                return f["url"]
    return None


def _last_updated_epoch(payload: dict) -> int:
    # Some providers expose GBFS-compliant `last_updated`, others add `lastUpdatedOther`.
    # Prefer GBFS key and gracefully fallback.
    for key in ("last_updated", "lastUpdatedOther"):
        v = payload.get(key)
        if isinstance(v, int) and v > 0:
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
    raise ValueError("Missing valid last_updated / lastUpdatedOther in GBFS payload")


def handler(event, context):
    root = GBFS_ROOT[CITY]
    # Prefer feed discovery via gbfs.json (works even if language path changes).
    gbfs = _get_json(root)
    status_url = _find_feed_url(gbfs, "station_status")
    info_url = _find_feed_url(gbfs, "station_information")

    if not status_url or not info_url:
        base = root.replace("/gbfs.json", "")
        status_url = status_url or f"{base}/station_status.json"
        info_url = info_url or f"{base}/station_information.json"

    status = _get_json(status_url)
    info = _get_json(info_url)

    # 校验（失败抛异常 -> 被捕获 -> 写错误日志）
    validate_station_status(status)
    validate_station_info(info)

    source_ts = _last_updated_epoch(status)
    dt_prefix = _dt_prefix_from_epoch(source_ts)
    status_key = f"raw/station_status/city={CITY}/{dt_prefix}/data.json.gz"
    info_key = f"raw/station_information/city={CITY}/{dt_prefix}/data.json.gz"

    # 幂等：若已存在则跳过
    if _exists(status_key) and _exists(info_key):
        return {"skipped": True, "status_key": status_key, "info_key": info_key}

    _put_json(status, status_key)
    _put_json(info, info_key)
    # 各自目录写 manifest（可选）
    _put_json(
        {"city": CITY, "source_ts": source_ts, "ingested_utc": int(time.time())},
        f"raw/station_status/city={CITY}/{dt_prefix}/_manifest.json.gz",
    )
    _put_json(
        {"city": CITY, "source_ts": source_ts, "ingested_utc": int(time.time())},
        f"raw/station_information/city={CITY}/{dt_prefix}/_manifest.json.gz",
    )
    return {"ok": True, "prefix": dt_prefix}


if __name__ == "__main__":

    BUCKET = os.getenv("BUCKET")
    CITY = os.getenv("CITY", "paris")
    if not BUCKET:
        raise RuntimeError("Env BUCKET is required, e.g. BUCKET=mlops-bikeshare-...")

    print(f"[gbfs_ingest] city={CITY}, bucket=s3://{BUCKET}")
    res = handler({}, None)
    print("[gbfs_ingest] result:", json.dumps(res))
