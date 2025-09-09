import gzip
import io
import json
import os
import time
from datetime import datetime, timedelta, timezone

import boto3
import requests

from .validators import validate_weather


def floor_to_5min(ts: datetime) -> datetime:
    ts = ts.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return ts.replace(minute=(ts.minute // 5) * 5)

BUCKET = os.getenv("BUCKET")
CITY   = os.getenv("CITY", "nyc")

# Provider selection: official | rapidapi
METEOSTAT_PROVIDER = os.getenv("METEOSTAT_PROVIDER", "rapidapi").lower().strip()
API_KEY = os.getenv("METEOSTAT_API_KEY", "").strip()

# City latitude and longitude (expandable)
CITY_COORDS = {
    "nyc":  (40.7128, -74.0060),
    "paris":(48.8566, 2.3522),
}

s3 = boto3.client("s3")

def _put_json(obj:dict, key:str):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue(),
                  ContentType="application/json", ContentEncoding="gzip")

def _fetch_meteostat(lat:float, lon:float, start:datetime, end:datetime)->dict:
    """
    Meteostat has two providers:
    - official: https://api.meteostat.net/v2/point/hourly (header: x-api-key)
    - rapidapi: https://meteostat.p.rapidapi.com/point/hourly
    (headers: x-rapidapi-key + x-rapidapi-host)
    Note: RapidAPI's point/hourly requires the start/end values ​​to be in YYYY-MM-DD format (pure date).
    """
    if not API_KEY:
        raise RuntimeError("Missing METEOSTAT_API_KEY")

    params = {
        "lat": f"{lat:.4f}",
        "lon": f"{lon:.4f}",
        "tz": "UTC",
    }

    if METEOSTAT_PROVIDER == "official":
        base = "https://api.meteostat.net/v2/point/hourly"
        params.update({
            "start": start.strftime("%Y-%m-%d %H:%M"),
            "end":   end.strftime("%Y-%m-%d %H:%M"),
        })
        headers = {"x-api-key": API_KEY}
    else:
        # rapidapi
        base = "https://meteostat.p.rapidapi.com/point/hourly"
        # Only accept YYYY-MM-DD
        params.update({
            "start": start.strftime("%Y-%m-%d"),
            "end":   end.strftime("%Y-%m-%d"),
        })
        headers = {
            "x-rapidapi-key": API_KEY,
            "x-rapidapi-host": "meteostat.p.rapidapi.com",
        }

    r = requests.get(base, params=params, headers=headers, timeout=15)
    if not r.ok:
        raise RuntimeError(f"Meteostat HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    validate_weather(data)
    # Compatible output: Make sure the data array contains at least the 
    # time/temp/prcp/wnd keys (some field names are different)
    rows = data.get("data", [])
    norm = []
    for row in rows:
        # Official return keys: time/temp/prcp/wspd, etc.; 
        # rapidapi will keep the same name
        raw_time = row.get("time") or row.get("date") or row.get("datetime")
        if raw_time:
            t = raw_time.replace("T", " ")
            t = t[:16]  # Only keep to the minute
        else:
            t = None
        norm.append({
            "time": t,  # 'YYYY-MM-DD HH:MM'
            "temp": row.get("temp") or row.get("temperature"),
            "prcp": row.get("prcp") or row.get("precipitation"),
            "wnd":  row.get("wspd") or row.get("wind_speed") or row.get("wind"),
            "rhum": row.get("rhum") or row.get("humidity"),
            "pres": row.get("pres") or row.get("pressure"),
            "wdir": row.get("wdir") or row.get("wind_direction"),
            "wpgt": row.get("wpgt") or row.get("wind_gust"),
            "snow": row.get("snow"),
            "coco": row.get("coco") or row.get("weather_code"),
        })
    return {"meta": {"source": f"meteostat-{METEOSTAT_PROVIDER}"}, "data": norm}

def _fetch_open_meteo(lat:float, lon:float, start:datetime, end:datetime)->dict:
    """
    兜底：Open‑Meteo（免 key），最小映射到 time/temp/prcp/wnd
    """
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
    rhums = hourly.get("relativehumidity_2m", []) or []
    press = hourly.get("surface_pressure", []) or []
    wdirs = hourly.get("winddirection_10m", []) or []
    gusts = hourly.get("windgusts_10m", []) or []
    rows = []
    for i, ts in enumerate(times):
        rows.append({
            "time": ts.replace("T", " "),
            "temp": temps[i] if i < len(temps) else None,
            "prcp": prcps[i] if i < len(prcps) else None,
            "wnd":  winds[i] if i < len(winds) else None,
            "rhum": rhums[i] if i < len(rhums) else None,
            "pres": press[i] if i < len(press) else None,
            "wdir": wdirs[i] if i < len(wdirs) else None,
            "wpgt": gusts[i] if i < len(gusts) else None,
            "snow": None,
            "coco": None,
        })
    data = {"meta": {"source": "open-meteo"}, "data": rows}
    validate_weather(data)
    return data

def handler(event, context):
    if not BUCKET:
        raise RuntimeError("Env BUCKET is required, e.g. BUCKET=mlops-bikeshare-...")

    lat, lon = CITY_COORDS[CITY]
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=2)
    end   = now

    # Try Meteostat first (automatically choose official/rapidapi based on provider); fallback to Open‑Meteo if that fails
    try:
        payload = _fetch_meteostat(lat, lon, start, end)
        source = payload.get("meta", {}).get("source", "meteostat")
    except Exception as e:
        print(f"[weather_ingest] Meteostat failed: {e}. Falling back to Open‑Meteo.")
        payload = _fetch_open_meteo(lat, lon, start, end)
        source = "open-meteo"

    src = payload.get("meta", {}).get("source", "")
    if "meteostat" not in src and "open-meteo" not in src:
        raise RuntimeError(f"[weather_ingest] unexpected weather source: {src}")

    dt5 = floor_to_5min(now)
    dt_prefix = dt5.strftime("dt=%Y-%m-%d-%H-%M")
    key = f"raw/weather_hourly/city={CITY}/{dt_prefix}/data.json.gz"

    payload["_meta_ingest"] = {"city": CITY, "source": source, "ingested_utc": int(time.time())}
    _put_json(payload, key)
    return {"ok": True, "key": key, "source": source}

if __name__ == "__main__":
    print(f"[weather_ingest] city={CITY}, bucket=s3://{BUCKET}, provider={METEOSTAT_PROVIDER}")
    res = handler({}, None)
    print("[weather_ingest] result:", json.dumps(res))
