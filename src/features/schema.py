import re
from typing import List

import pandas as pd

FEATURE_COLUMNS: List[str] = [
    "util_bikes","util_docks",
    "delta_bikes_5m","delta_docks_5m",
    "roll15_net_bikes","roll30_net_bikes","roll60_net_bikes",
    "roll15_bikes_mean","roll30_bikes_mean","roll60_bikes_mean",
    "nbr_bikes_weighted","nbr_docks_weighted",
    "hour","dow","is_weekend","is_holiday",
    "temp_c","precip_mm","wind_kph",
    "rhum_pct","pres_hpa","wind_dir_deg","wind_gust_kph","snow_mm","weather_code"
]

LABEL_COLUMNS: List[str] = [
    "y_stockout_bikes_30","y_stockout_docks_30",
    "target_bikes_t30","target_docks_t30"
]

REQUIRED_BASE = ["city","dt","station_id","capacity","lat","lon","bikes","docks"]

def _is_dt_string(s: pd.Series) ->bool:
    pat = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}")
    return s.dropna().map(lambda x: bool(pat.match(str(x)))).all()

def validate_feature_df(df: pd.DataFrame, missing_rate_threshold: float = 0.01):
    # 1. basic columns completed
    for c in REQUIRED_BASE + FEATURE_COLUMNS + LABEL_COLUMNS:
        if c not in df.columns:
            raise  ValueError(f"missing expected columns: {c}")
    
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
        

 
    # --- NEW STEP: drop all-NaN feature columns ---
    feat = df[FEATURE_COLUMNS].copy()
    all_nan_cols = [c for c in feat.columns if feat[c].isna().all()]
    if all_nan_cols:
        print(f"[INFO] Dropping {len(all_nan_cols)} all-NaN feature columns: {all_nan_cols}")
        feat = feat.drop(columns=all_nan_cols)

    # 3. The overall feature missing rate is < threshold
    miss = feat.isna().mean().mean()
    if miss > missing_rate_threshold:
        raise ValueError(f"feature missing rate {miss:.2%} > {missing_rate_threshold:.2%}")
    return True