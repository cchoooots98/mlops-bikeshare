# Training & Evaluation - Step 4

## Objectives
- Train a first binary classifier to predict 30-minute stockout risk per station.
- Keep experiments traceable in MLflow.
- Maintain an architecture that can migrate cleanly to dbt-owned production feature tables.

## Data and Labels
- Target source table: `analytics.feat_station_snapshot_5min` in Postgres/dbt.
- Current repository state: the Python training code still reads `features_offline` (Athena external). Migrating training to dbt-owned feature tables is planned but not implemented in this phase.
- Feature set: defined centrally in `schema.py -> FEATURE_COLUMNS`.
- Labels:
  - `y_stockout_bikes_30`
  - `y_stockout_docks_30`
- Numeric targets:
  - `target_bikes_t30`
  - `target_docks_t30`

## Direction of Travel
- dbt is the long-term owner of production feature generation.
- Python training will later become a consumer of dbt feature tables instead of an Athena-built offline table.
- The long-term weather feature contract follows `dim_weather`, not the legacy Athena weather schema.

## Weather Feature Direction
Target weather feature columns:

- `temperature_c`
- `humidity_pct`
- `wind_speed_ms`
- `precipitation_mm`
- `weather_code`
- `hourly_temperature_c`
- `hourly_humidity_pct`
- `hourly_wind_speed_ms`
- `hourly_precipitation_mm`
- `hourly_precipitation_probability_pct`
- `hourly_weather_code`

Deprecated weather feature columns:

- `temp_c`
- `precip_mm`
- `wind_kph`
- `rhum_pct`
- `pres_hpa`
- `wind_dir_deg`
- `wind_gust_kph`
- `snow_mm`

## Temporal Split
- Train on earlier timestamps and validate on later timestamps.
- Keep a gap window between train and validation slices to reduce leakage from rolling features.

## Metrics
- Primary metric: PR-AUC on validation.
- Overfitting check: gap between train PR-AUC and validation PR-AUC.
- Threshold selection: optimize on validation, not on training data.

## Current Execution Note
The current code path and CLI examples still reflect the existing Athena-based training implementation. Those commands remain valid until the later feature-layer migration is implemented.
