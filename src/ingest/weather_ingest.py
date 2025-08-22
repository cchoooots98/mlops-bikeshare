import os, json, gzip, io, time
from datetime import datetime, timedelta, timezone
from typing import Tuple
import requests, boto3
from .validators import validate_weather

def floor_to_5min(ts: datetime) -> datetime:
    ts = ts.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return ts.replace(minute=(ts.minute // 5) * 5)


BUCKET = os.getenv("BUCKET")
CITY   = os.getenv("CITY", "nyc")
API_KEY = os.getenv("METEOSTAT_API_KEY", "").strip()

# 城市经纬度（可根据 station_information 做每日更新，这里先用中心点）
CITY_COORDS = {
    "nyc":  (40.7128, -74.0060),
    "paris":(48.8566, 2.3522),
}

s3 = boto3.client("s3")

def _utc_floor_minute(dt:datetime)->datetime:
    return dt.replace(second=0, microsecond=0)

def _put_json(obj:dict, key:str):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(json.dumps(obj).encode("utf-8"))
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue(),
                  ContentType="application/json", ContentEncoding="gzip")

def _fetch_meteostat(lat:float, lon:float, start:datetime, end:datetime)->dict:
    """
    调用 Meteostat v2: https://api.meteostat.net/v2/point/hourly
    需要在请求头里携带 x-api-key。时间用 'YYYY-MM-DD HH:MM'（UTC）。
    """
    if not API_KEY:
        raise RuntimeError("Missing METEOSTAT_API_KEY for Meteostat v2 API")

    base = "https://api.meteostat.net/v2/point/hourly"
    params = {
        "lat": f"{lat:.4f}",
        "lon": f"{lon:.4f}",
        "start": start.strftime("%Y-%m-%d %H:%M"),
        "end":   end.strftime("%Y-%m-%d %H:%M"),
        "tz": "UTC",
    }
    headers = {"x-api-key": API_KEY}
    r = requests.get(base, params=params, headers=headers, timeout=15)
    if not r.ok:
        # 打印前 300 个字符帮助排障
        raise RuntimeError(f"Meteostat HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    # 轻校验
    # MeteostatPayload(**data)
    validate_weather(data)
    return data

def _fetch_open_meteo(lat:float, lon:float, start:datetime, end:datetime)->dict:
    """
    兜底：Open‑Meteo（免 key），返回结构与 Meteostat 不同，这里做一个最小映射：
    仅保留 time/temp/prcp/wind_speed 这些与特征工程相关的字段。
    """
    # Open‑Meteo 按小时给回 arrays；我们将其转为 MeteostatPayload 兼容结构
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "timezone": "UTC",
        "start_hour": start.strftime("%Y-%m-%dT%H:00"),
        "end_hour":   end.strftime("%Y-%m-%dT%H:00"),
    }
    r = requests.get(base, params=params, timeout=15)
    if not r.ok:
        raise RuntimeError(f"Open‑Meteo HTTP {r.status_code}: {r.text[:300]}")
    j = r.json()
    hourly = j.get("hourly", {})
    times = hourly.get("time", []) or []
    temps = hourly.get("temperature_2m", []) or []
    prcps = hourly.get("precipitation", []) or []
    winds = hourly.get("wind_speed_10m", []) or []
    rows = []
    for i, ts in enumerate(times):
        rows.append({
            "time": ts.replace("T", " "),      # 兼容 MeteostatPayload 的 datetime 解析
            "temp": temps[i] if i < len(temps) else None,
            "prcp": prcps[i] if i < len(prcps) else None,
            "wnd":  winds[i] if i < len(winds) else None,
        })
    data = {"meta": {"source": "open-meteo"}, "data": rows}
    # 仍然用你的 Pydantic 模型做轻校验
    #MeteostatPayload(**data)
    validate_weather(data)
    return data

def handler(event, context):
    if not BUCKET:
        raise RuntimeError("Env BUCKET is required, e.g. BUCKET=mlops-bikeshare-...")

    lat, lon = CITY_COORDS[CITY]
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=2)
    end   = now

    # 先尝试 Meteostat（有 key）；失败则兜底 Open‑Meteo（免 key）
    try:
        payload = _fetch_meteostat(lat, lon, start, end)
        source = "meteostat"
    except Exception as e:
        # 打印告警但不中断；降级到 Open‑Meteo
        print(f"[weather_ingest] Meteostat failed: {e}. Falling back to Open‑Meteo.")
        payload = _fetch_open_meteo(lat, lon, start, end)
        source = "open-meteo"
    dt5 = floor_to_5min(now)
    dt_prefix = dt5.strftime("dt=%Y-%m-%d-%H-%M")
    key = f"raw/city={CITY}/{dt_prefix}/weather_point_hourly.json.gz"
    # 附加元数据
    payload["_meta_ingest"] = {"city": CITY, "source": source, "ingested_utc": int(time.time())}

    _put_json(payload, key)
    return {"ok": True, "key": key, "source": source}

if __name__ == "__main__":
    print(f"[weather_ingest] city={CITY}, bucket=s3://{BUCKET}")
    res = handler({}, None)
    print("[weather_ingest] result:", json.dumps(res))
