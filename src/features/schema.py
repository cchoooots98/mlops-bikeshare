import re
from typing import List

import pandas as pd

# dbt feature tables are the formal producer contract; Python validates and consumes that contract.
FEATURE_COLUMNS: List[str] = [
    "util_bikes",
    "util_docks",
    "delta_bikes_5m",
    "delta_docks_5m",
    "roll15_net_bikes",
    "roll30_net_bikes",
    "roll60_net_bikes",
    "roll15_bikes_mean",
    "roll30_bikes_mean",
    "roll60_bikes_mean",
    "nbr_bikes_weighted",
    "nbr_docks_weighted",
    "has_neighbors_within_radius",
    "neighbor_count_within_radius",
    "minutes_since_prev_snapshot",
    "hour",
    "dow",
    "is_weekend",
    "is_holiday",
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms",
    "precipitation_mm",
    "weather_code",
    "hourly_temperature_c",
    "hourly_humidity_pct",
    "hourly_wind_speed_ms",
    "hourly_precipitation_mm",
    "hourly_precipitation_probability_pct",
    "hourly_weather_code",
]

LABEL_COLUMNS: List[str] = ["y_stockout_bikes_30", "y_stockout_docks_30", "target_bikes_t30", "target_docks_t30"]

REQUIRED_BASE = ["city", "dt", "station_id", "capacity", "lat", "lon", "bikes", "docks"]
NULLABLE_BY_DESIGN_FEATURE_COLUMNS = ["nbr_bikes_weighted", "nbr_docks_weighted"]


def _is_dt_string(s: pd.Series) -> bool:
    pat = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}")
    return s.dropna().map(lambda x: bool(pat.match(str(x)))).all()


def validate_feature_df(df: pd.DataFrame, missing_rate_threshold: float = 0.01):
    # 1. basic columns completed
    for c in REQUIRED_BASE + FEATURE_COLUMNS + LABEL_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"missing expected columns: {c}")

    # 2. basic type and value checking (lightweight)
    if not _is_dt_string(df["dt"]):
        raise ValueError(" dt must be 'YYYY-MM-DD-HH-mm' string")

    if (df["capacity"] < 0).any():
        raise ValueError("capacity must be >=0")

    # utilization must be in [0,1] (ignore NaNs)
    for c in ["util_bikes", "util_docks"]:
        s = pd.to_numeric(df[c], errors="coerce")
        mask = s.isna() | ((s >= 0) & (s <= 1))
        if not mask.all():
            bad = int((~mask).sum())
            raise ValueError(f"{c} must be in [0,1]; {bad} values out of range")

    neighbor_flag = pd.to_numeric(df["has_neighbors_within_radius"], errors="coerce")
    neighbor_flag_mask = neighbor_flag.isna() | neighbor_flag.isin([0, 1])
    if not neighbor_flag_mask.all():
        bad = int((~neighbor_flag_mask).sum())
        raise ValueError(f"has_neighbors_within_radius must be 0/1; {bad} values out of range")

    neighbor_count = pd.to_numeric(df["neighbor_count_within_radius"], errors="coerce")
    neighbor_count_mask = neighbor_count.isna() | (neighbor_count >= 0)
    if not neighbor_count_mask.all():
        bad = int((~neighbor_count_mask).sum())
        raise ValueError(f"neighbor_count_within_radius must be >=0; {bad} values out of range")

    gap_minutes = pd.to_numeric(df["minutes_since_prev_snapshot"], errors="coerce")
    gap_minutes_mask = gap_minutes.isna() | (gap_minutes >= 0)
    if not gap_minutes_mask.all():
        bad = int((~gap_minutes_mask).sum())
        raise ValueError(f"minutes_since_prev_snapshot must be >=0; {bad} values out of range")

    # --- NEW STEP: drop all-NaN feature columns ---
    feat = df[FEATURE_COLUMNS].copy()
    for c in NULLABLE_BY_DESIGN_FEATURE_COLUMNS:
        if c in feat.columns:
            feat = feat.drop(columns=[c])
    all_nan_cols = [c for c in feat.columns if feat[c].isna().all()]
    if all_nan_cols:
        print(f"[INFO] Dropping {len(all_nan_cols)} all-NaN feature columns: {all_nan_cols}")
        feat = feat.drop(columns=all_nan_cols)

    # 3. The overall feature missing rate is < threshold
    miss = feat.isna().mean().mean()
    if miss > missing_rate_threshold:
        raise ValueError(f"feature missing rate {miss:.2%} > {missing_rate_threshold:.2%}")
    return True
