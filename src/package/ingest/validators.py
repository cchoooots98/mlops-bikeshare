from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime, timezone

# ---- GBFS station_status ----
class StationStatusItem(BaseModel):
    station_id: str
    num_bikes_available: int = Field(ge=0, le=2000)
    num_docks_available: int = Field(ge=0, le=2000)
    is_installed: Optional[int] = Field(None, ge=0, le=1)
    is_renting: Optional[int] = Field(None, ge=0, le=1)
    is_returning: Optional[int] = Field(None, ge=0, le=1)
    last_reported: int = Field(ge=1)

class StationStatusPayload(BaseModel):
    last_updated: int
    data: dict

    @field_validator("data")
    @classmethod
    def validate_data(cls, v):
        stations = v.get("stations", [])
        # 缺失率与数量下限（拒绝过小批次）
        assert isinstance(stations, list) and len(stations) >= 10
        _ = [StationStatusItem(**s) for s in stations]
        return v

# ---- GBFS station_information ----
class StationInfoItem(BaseModel):
    station_id: str
    name: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    capacity: Optional[int] = Field(None, ge=0, le=500)

class StationInfoPayload(BaseModel):
    data: dict
    @field_validator("data")
    @classmethod
    def validate_data(cls, v):
        stations = v.get("stations", [])
        assert isinstance(stations, list) and len(stations) >= 10
        _ = [StationInfoItem(**s) for s in stations]
        return v

# ---- Meteostat point hourly (精简契约) ----
class MeteostatHour(BaseModel):
    time: datetime
    temp: Optional[float] = Field(None, ge=-60, le=60)
    prcp: Optional[float] = Field(None, ge=0, le=500)
    wnd: Optional[float] = Field(None, ge=0, le=150)

class MeteostatPayload(BaseModel):
    meta: dict
    data: List[MeteostatHour]
