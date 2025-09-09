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
CITY = os.getenv("CITY", "nyc")

GBFS_ROOT = {
    "nyc": "https://gbfs.citibikenyc.com/gbfs/en",
    "paris": "https://velib-metropole-opendata.smoove.pro/gbfs/gbfs.json",
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


def handler(event, context):
    root = GBFS_ROOT[CITY]
    # NYC 直接给 en 路径；其他城市可能需要先读 gbfs.json 解析子链接
    if CITY == "nyc":
        status_url = f"{root}/station_status.json"
        info_url = f"{root}/station_information.json"
    else:
        # 简化：直接拼典型文件名；必要时先 requests.get(root) 再解析 feeds
        status_url = root.replace("gbfs.json", "en/station_status.json")
        info_url = root.replace("gbfs.json", "en/station_information.json")

    status = requests.get(status_url, timeout=10).json()
    info = requests.get(info_url, timeout=10).json()

    # 校验（失败抛异常 -> 被捕获 -> 写错误日志）
    validate_station_status(status)
    validate_station_info(info)

    dt_prefix = _dt_prefix_from_epoch(status["last_updated"])
    status_key = f"raw/station_status/city={CITY}/{dt_prefix}/data.json.gz"
    info_key = f"raw/station_information/city={CITY}/{dt_prefix}/data.json.gz"

    # 幂等：若已存在则跳过
    if _exists(status_key) and _exists(info_key):
        return {"skipped": True, "status_key": status_key, "info_key": info_key}

    _put_json(status, status_key)
    _put_json(info, info_key)
    # 各自目录写 manifest（可选）
    _put_json(
        {"city": CITY, "source_ts": status["last_updated"], "ingested_utc": int(time.time())},
        f"raw/station_status/city={CITY}/{dt_prefix}/_manifest.json.gz",
    )
    _put_json(
        {"city": CITY, "source_ts": status["last_updated"], "ingested_utc": int(time.time())},
        f"raw/station_information/city={CITY}/{dt_prefix}/_manifest.json.gz",
    )
    return {"ok": True, "prefix": dt_prefix}


if __name__ == "__main__":

    BUCKET = os.getenv("BUCKET")
    CITY = os.getenv("CITY", "nyc")
    if not BUCKET:
        raise RuntimeError("Env BUCKET is required, e.g. BUCKET=mlops-bikeshare-...")

    print(f"[gbfs_ingest] city={CITY}, bucket=s3://{BUCKET}")
    res = handler({}, None)
    print("[gbfs_ingest] result:", json.dumps(res))
