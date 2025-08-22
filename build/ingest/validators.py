# src/validators.py
from typing import Dict, List

# --- 首选：尝试使用 Pydantic v2 ---
try:
    from pydantic import BaseModel, Field

    class StationStatusItem(BaseModel):
        station_id: str
        num_bikes_available: int = Field(ge=0, le=2000)
        num_docks_available: int = Field(ge=0, le=2000)
        last_reported: int = Field(ge=1)

    class StationStatusPayload(BaseModel):
        last_updated: int
        data: dict

    def validate_station_status(payload: Dict):
        stations = payload.get("data", {}).get("stations", [])
        _ = [StationStatusItem(**s) for s in stations]
        if len(stations) < 10:
            raise ValueError("stations too few")

    class StationInfoItem(BaseModel):
        station_id: str
        name: str
        lat: float
        lon: float
        capacity: int | None = None

    def validate_station_info(payload: Dict):
        stations = payload.get("data", {}).get("stations", [])
        _ = [StationInfoItem(**s) for s in stations]
        if len(stations) < 10:
            raise ValueError("stations too few")

    def validate_weather(payload: Dict):
        # 轻校验：至少存在 data 列表
        if not isinstance(payload.get("data", []), list):
            raise ValueError("weather.data must be list")

except Exception:
    # --- 没装到 pydantic 时：降级为纯 Python 校验 ---
    def validate_station_status(payload: Dict):
        stations = payload.get("data", {}).get("stations", [])
        if not isinstance(stations, list) or len(stations) < 10:
            raise ValueError("stations missing/too few")
        for s in stations:
            if "station_id" not in s:
                raise ValueError("missing station_id")
            nb = int(s.get("num_bikes_available", -1))
            nd = int(s.get("num_docks_available", -1))
            if not (0 <= nb <= 2000 and 0 <= nd <= 2000):
                raise ValueError("counts out of range")
            if int(s.get("last_reported", 0)) <= 0:
                raise ValueError("invalid last_reported")

    def validate_station_info(payload: Dict):
        stations = payload.get("data", {}).get("stations", [])
        if not isinstance(stations, list) or len(stations) < 10:
            raise ValueError("stations missing/too few")
        for s in stations:
            if "station_id" not in s or "name" not in s:
                raise ValueError("missing required fields")
            if "lat" not in s or "lon" not in s:
                raise ValueError("missing lat/lon")

    def validate_weather(payload: Dict):
        if not isinstance(payload.get("data", []), list):
            raise ValueError("weather.data must be list")
