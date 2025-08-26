# Feature Store Documentation

## 1. Alignment Between Offline and Online
- **Time granularity:** All features aligned on 5-minute `dt`.
- **Window definitions:** Same rolling windows (15/30/60 minutes) used offline and online.
- **Thresholds:** Stockout threshold = 2 bikes/docks.
- **Weather alignment:** Hourly weather aligned to station `dt` with forward-fill up to 6 hours.
- **Neighbor strategy:** K=5 nearest neighbors within radius 0.8 km, weighted by `1/distance`.

This guarantees that online featurization (`inference/featurize_online.py`) produces the same feature shapes as the offline pipeline.

---

## 2. Feature Definitions

| Field | Type | Description | Window / Rule | Online Availability |
|-------|------|-------------|----------------|---------------------|
| util_bikes | float [0,1] | Ratio of bikes/capacity | current | ✅ |
| util_docks | float [0,1] | Ratio of docks/capacity | current | ✅ |
| delta_bikes_5m | int | Change in bikes over 5 minutes | lag | ✅ |
| delta_docks_5m | int | Change in docks over 5 minutes | lag | ✅ |
| roll15_net_bikes | float | Net bikes outflow, 15-min rolling | rolling | ✅ |
| roll30_net_bikes | float | Net bikes outflow, 30-min rolling | rolling | ✅ |
| roll60_net_bikes | float | Net bikes outflow, 60-min rolling | rolling | ✅ |
| roll15_bikes_mean | float | Avg. available bikes (15-min) | rolling | ✅ |
| roll30_bikes_mean | float | Avg. available bikes (30-min) | rolling | ✅ |
| roll60_bikes_mean | float | Avg. available bikes (60-min) | rolling | ✅ |
| nbr_bikes_weighted | float | Neighbor bike availability, weighted | neighbors (K=5, r=0.8 km) | ✅ |
| nbr_docks_weighted | float | Neighbor dock availability, weighted | neighbors | ✅ |
| hour | int [0–23] | Hour of day | derived from dt | ✅ |
| dow | int [0–6] | Day of week | derived from dt | ✅ |
| is_weekend | bool | Weekend flag | derived | ✅ |
| is_holiday | bool | Holiday flag (pre-loaded calendar) | derived | ✅ |
| temp_c | float | Temperature in Celsius | hourly weather (fwd-fill 6h) | ✅ |
| precip_mm | float | Precipitation (mm) | hourly weather | ✅ |
| wind_kph | float | Wind speed (kph) | hourly weather | ✅ |
| rhum_pct | float | Relative humidity (%) | hourly weather | ✅ |
| pres_hpa | float | Pressure (hPa) | hourly weather | ✅ |
| wind_dir_deg | float | Wind direction (deg) | hourly weather | ✅ |
| wind_gust_kph | float | Wind gust (kph) | hourly weather | ✅ |
| snow_mm | float | Snow depth/precip (mm) | hourly weather | ✅ |
| weather_code | int | Weather condition code (Meteostat `coco`) | hourly weather | ✅ |

---

## 3. Label Definitions

- **Binary labels (classification):**
  - `y_stockout_bikes_30` = 1 if bikes ≤ 2 at `t+30min`.
  - `y_stockout_docks_30` = 1 if docks ≤ 2 at `t+30min`.

- **Regression targets:**
  - `target_bikes_t30` = bikes available at `t+30min`.
  - `target_docks_t30` = docks available at `t+30min`.

---

## 4. Versioning and Reproducibility
- Always recompute with **latest `station_information` partition**.
- Features and labels stored under `s3://.../features/city=.../dt=YYYY-MM-DD-HH-mm/`.
- Re-running the same window range with fixed code produces reproducible features.
