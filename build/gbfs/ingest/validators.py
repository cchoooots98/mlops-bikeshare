# src/ingest/validators.py
from typing import Dict, List, Any
from datetime import datetime, timezone

def _as_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def _as_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _parse_utc_hour(s: str):
    s = s.replace("T", " ")
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    
    
# ---------- GBFS: station_status ----------
def validate_station_status(payload: Dict, *,
                            min_stations: int = 10,
                            max_count: int = 2000,
                            max_report_lag_hours: int = 6) -> None:
    stations = payload.get("data", {}).get("stations", [])
    if not isinstance(stations, list) or len(stations) < min_stations:
        raise ValueError("status.data.stations missing/too few")

    last_updated = _as_int(payload.get("last_updated"), -1)
    if last_updated <= 0:
        raise ValueError("status.last_updated invalid")

    seen = set()
    bad_lag = 0
    for s in stations:
        sid = s.get("station_id")
        if not sid:
            raise ValueError("status: missing station_id")
        if sid in seen:
            raise ValueError(f"status: duplicated station_id {sid}")
        seen.add(sid)

        nb = _as_int(s.get("num_bikes_available"), -1)
        nd = _as_int(s.get("num_docks_available"), -1)
        lr = _as_int(s.get("last_reported"), 0)

        if not (0 <= nb <= max_count and 0 <= nd <= max_count):
            raise ValueError("status: counts out of range")

        if lr <= 0:
            raise ValueError("status: invalid last_reported")

        # 抓包时间与站点上报时间差，过大则计数（不直接报错，给上游观测）
        if last_updated - lr > max_report_lag_hours * 3600:
            bad_lag += 1

    # 若滞后站点比例过大，可以直接拒绝，避免整批失真
    if bad_lag / max(len(stations), 1) > 0.2:
        raise ValueError("status: too many stale stations (>20%)")

# ---------- GBFS: station_information ----------
def validate_station_info(payload: Dict, *,
                          min_stations: int = 10) -> None:
    stations = payload.get("data", {}).get("stations", [])
    if not isinstance(stations, list) or len(stations) < min_stations:
        raise ValueError("info.data.stations missing/too few")

    seen = set()
    for s in stations:
        sid = s.get("station_id")
        name = s.get("name")
        lat = _as_float(s.get("lat"))
        lon = _as_float(s.get("lon"))
        cap = s.get("capacity", None)
        cap = None if cap is None else _as_int(cap, -1)

        if not sid:
            raise ValueError("info: missing station_id")
        if sid in seen:
            raise ValueError(f"info: duplicated station_id {sid}")
        seen.add(sid)

        if lat is None or lon is None:
            raise ValueError("info: missing lat/lon")
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise ValueError("info: lat/lon out of range")

        if cap is not None and cap < 0:
            raise ValueError("info: capacity < 0")

# ---------- Weather (Meteostat/Open‑Meteo mapped) ----------
def validate_weather(payload: Dict, *,
                     min_rows: int = 1,
                     temp_range=(-50.0, 60.0),
                     wind_range=(0.0, 200.0),
                     prcp_range=(0.0, 500.0)) -> None:
    rows = payload.get("data", [])
    if not isinstance(rows, list):
        raise ValueError("weather.data must be list")
    if len(rows) < min_rows:
        raise ValueError("weather.data too few")

    prev_ts = None
    for r in rows:
        t = r.get("time")
        if not t:
            raise ValueError("weather: missing time")
        ts = _parse_utc_hour(t)

        # 简单单调递增与去重检查
        if prev_ts and ts <= prev_ts:
            raise ValueError("weather: time not strictly increasing")
        prev_ts = ts

        # 字段存在即可，数值允许缺失，但若给了就做范围校验
        for key, lo, hi in [
            ("temp", *temp_range),
            ("wnd", *wind_range),
            ("prcp", *prcp_range),
        ]:
            v = r.get(key, None)
            if v is None:
                continue
            fv = _as_float(v, None)
            if fv is None or not (lo <= fv <= hi):
                raise ValueError(f"weather: {key} out of range or invalid")
