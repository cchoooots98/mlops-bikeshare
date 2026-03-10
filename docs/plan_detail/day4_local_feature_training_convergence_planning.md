# Day 4 Guide: Local Feature, Training, and Online Contract Convergence

This Day 4 guide continues the sequence from Day 1 to Day 3:

1. finish the dbt feature layer
2. remove Python and Athena from the formal feature path
3. make training read Postgres feature tables
4. make online featurization and prediction consume the same dbt contract
5. define the local MLflow Registry standard for later AWS promotion

Day 4 is where the project stops being "Python feature scripts plus some dbt marts"
and becomes a dbt-first local ML platform.

---

## 1. What you should finish by the end of Day 4

By the end of Day 4, you should have:

- dbt models for the full local feature path
- a final offline feature table built in Postgres
- a final latest-online feature table built in Postgres
- `train.py` reading Postgres instead of Athena
- `featurize_online.py` reading dbt latest features instead of rebuilding features in Python
- prediction input columns aligned to the final dbt feature contract
- a documented MLflow Registry naming standard for local candidate models

What Day 4 should not try to finish yet:

- SageMaker endpoint deployment
- AWS blue/green or canary rollout
- production admission gates
- final dashboard migration

Those come after the local contract is stable.

---

## 2. Starting assumption for this guide

Assume Day 3 is already complete:

- local Docker services can start
- `dw-postgres` is reachable on `localhost:15432`
- Airflow DAGs can populate:
  - `public.stg_station_information`
  - `public.stg_station_status`
  - `public.stg_weather_current`
  - `public.stg_weather_hourly`
  - `public.stg_holidays`
- dbt can build:
  - `analytics.dim_weather`
  - `analytics.dim_date`

This guide explains what Day 4 adds on top of that Day 3 baseline.

---

## 3. Before coding Day 4

Run this first in PowerShell:

```powershell
. .\.venv\Scripts\Activate.ps1

# If AWS auth is still needed for raw-ingest validation
aws sso login --profile Shirley-fr

# Start local services
docker compose up airflow-init
docker compose up -d airflow-postgres dw-postgres airflow-webserver airflow-scheduler

# Sanity checks
docker compose ps
docker compose exec airflow-webserver airflow dags list
docker compose exec airflow-webserver airflow dags list-import-errors

# Confirm source staging and existing marts exist
docker compose exec dw-postgres psql -U velib -d velib_dw -c "\dt public.*"
docker compose exec dw-postgres psql -U velib -d velib_dw -c "\dt analytics.*"

# Confirm dbt still connects
dbt debug --project-dir dbt\bikeshare_dbt --profiles-dir dbt
dbt parse --project-dir dbt\bikeshare_dbt --profiles-dir dbt
dbt ls --project-dir dbt\bikeshare_dbt --profiles-dir dbt
```

If any of those fail, fix that first.
Do not start Day 4 model work on top of a broken Day 3 base.

---

## 4. Understand the current gap before changing code

Current repository reality:

- dbt currently owns only part of the warehouse:
  - `dim_weather`
  - `dim_date`
- Python still owns the real feature logic in:
  - [build_features.py](/c:/Career/selfGrowth/projects/mlops-bikeshare-202508/src/features/build_features.py)
- online features still rebuild logic in Python in:
  - [featurize_online.py](/c:/Career/selfGrowth/projects/mlops-bikeshare-202508/src/inference/featurize_online.py)
- training still reads Athena in:
  - [train.py](/c:/Career/selfGrowth/projects/mlops-bikeshare-202508/src/training/train.py)
- the canonical Python feature list still lives in:
  - [schema.py](/c:/Career/selfGrowth/projects/mlops-bikeshare-202508/src/features/schema.py)

This means the project still has two different truths:

1. dbt truth for dimensions
2. Python truth for features

Day 4 exists to remove that split.

---

## 5. The most important Day 4 design decisions

Before touching files, lock these decisions.

### 5.1 dbt becomes the only formal feature owner

From Day 4 onward:

- Python ingestion owns `public.stg_*`
- dbt owns all formal feature engineering
- Python training and inference only consume final dbt outputs

This is the core architecture correction.

### 5.2 Make the ML weather contract follow `dim_weather`

Current `FEATURE_COLUMNS` in [schema.py](/c:/Career/selfGrowth/projects/mlops-bikeshare-202508/src/features/schema.py)
still expect legacy weather-style names, but Day 4 should not preserve that weather naming as the final target.

Current warehouse weather contract in `dim_weather` is:

- `temperature_c`
- `humidity_pct`
- `wind_speed_ms`
- `precipitation_mm`
- `weather_code`
- `weather_main`
- `hourly_temperature_c`
- `hourly_humidity_pct`
- `hourly_wind_speed_ms`
- `hourly_precipitation_mm`
- `hourly_precipitation_probability_pct`
- `hourly_weather_code`
- `hourly_weather_main`

Recommended Day 4 trade-off:

- make ML weather columns align to `dim_weather`, not to legacy Python weather names
- update Python consumers to follow the dbt / `dim_weather` weather contract
- remove the old weather compatibility layer instead of carrying it into the formal feature tables

Why this is the best Day 4 choice:

- it makes `dim_weather` the actual weather source of truth instead of only a mart-level side contract
- it prevents dbt feature models from re-introducing deprecated weather naming just to preserve Python inertia
- it keeps one clear weather contract across offline training, online prediction, and model validation

What this means in practice:

- `dim_weather` remains the canonical warehouse weather mart
- `feat_station_snapshot_5min` and `feat_station_snapshot_latest` expose weather fields using the same ML-facing names as `dim_weather`
- `schema.py`, `train.py`, `featurize_online.py`, and `predictor.py` must be revised to consume that contract

This is the right engineering trade-off for Day 4 because the purpose of Day 4 is contract convergence, not compatibility preservation.

### 5.2.1 Weather contract boundary for Day 4

`dim_weather` should continue to retain a richer mart contract than the final ML input schema.

Keep these weather columns in `dim_weather` as the ML-facing weather subset:

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

Keep these columns in `dim_weather` for audit, analysis, or cleaning, but do not include them in final training or prediction inputs:

- `weather_main`
- `hourly_weather_main`
- `weather_description`
- `hourly_forecast_at`
- `source`
- `snapshot_bucket_at_utc`

This boundary should be explicit in every Day 4 feature-layer section that references weather columns.

### 5.3 Separate offline and online feature outputs in dbt

Recommended final Day 4 dbt feature outputs:

- `analytics.feat_station_snapshot_5min`
- `analytics.feat_station_snapshot_latest`

Recommended roles:

- `feat_station_snapshot_5min`
  - one row per `city + station_id + snapshot timestamp`
  - used by training and backtesting
- `feat_station_snapshot_latest`
  - latest row per `city + station_id`
  - used by online prediction

Why not calculate "latest" in Python:

- because then the online path becomes a different logic path from training
- because latest-row selection is easy and stable in SQL
- because one source of truth is the point of Day 4

### 5.4 Postgres is the local serving truth for Day 4

Day 4 should not keep Athena in the formal loop.

That means:

- `train.py` reads Postgres
- `featurize_online.py` reads Postgres
- predictor consumes Postgres-derived features

Athena can remain only as legacy or optional analysis support.

### 5.5 MLflow naming must become task-specific now

Current local scripts still allow generic names like:

- `bikeshare_risk`

That is too vague once you have:

- bike shortage vs dock shortage
- xgboost vs lightgbm
- candidate vs champion
- later local vs staging vs prod

Recommended Day 4 naming convention:

- registry name:
  - `paris_y_stockout_bikes_30_xgboost`
  - `paris_y_stockout_docks_30_xgboost`
- run name:
  - `train-xgboost-y_stockout_bikes_30`
- model aliases or tags:
  - `candidate`
  - `champion`
  - `local_validated`

If your MLflow version does not support aliases cleanly, use tags and keep stage changes minimal for local work.

---

## 6. Files Day 4 should create or modify

### 6.1 dbt files

Create or modify these dbt files:

- create `dbt/bikeshare_dbt/models/marts/dim_station.sql`
- create `dbt/bikeshare_dbt/models/marts/dim_time.sql`
- create `dbt/bikeshare_dbt/models/marts/fct_station_status.sql`
- create `dbt/bikeshare_dbt/models/intermediate/int_station_neighbors.sql`
- create `dbt/bikeshare_dbt/models/intermediate/int_station_status_enriched.sql`
- create `dbt/bikeshare_dbt/models/features/feat_station_snapshot_5min.sql`
- create `dbt/bikeshare_dbt/models/features/feat_station_snapshot_latest.sql`
- create or extend schema YAML for marts, intermediate, and features
- update [dbt_project.yml](/c:/Career/selfGrowth/projects/mlops-bikeshare-202508/dbt/bikeshare_dbt/dbt_project.yml) so `intermediate` and `features` are defined

### 6.2 Python files

Modify these Python files:

- [schema.py](/c:/Career/selfGrowth/projects/mlops-bikeshare-202508/src/features/schema.py)
- [train.py](/c:/Career/selfGrowth/projects/mlops-bikeshare-202508/src/training/train.py)
- [featurize_online.py](/c:/Career/selfGrowth/projects/mlops-bikeshare-202508/src/inference/featurize_online.py)
- [predictor.py](/c:/Career/selfGrowth/projects/mlops-bikeshare-202508/src/inference/predictor.py)
- [register_model.py](/c:/Career/selfGrowth/projects/mlops-bikeshare-202508/pipelines/register_model.py)
- [promote.py](/c:/Career/selfGrowth/projects/mlops-bikeshare-202508/pipelines/promote.py)

Day 4 should not modify ingestion file ownership.

---

## 7. Build the missing dbt marts first

Do not start with training or predictor changes.
Finish the warehouse contract first.

### 7.1 Build `dim_station`

Purpose:

- one stable station row per business station
- the correct source for capacity, latitude, longitude, and station name

Recommended logic:

- treat `city + station_id` as the durable station business key
- build `dim_station` as an SCD2 dimension from station information staging
- track changes in:
  - `station_name`
  - `capacity`
  - `latitude`
  - `longitude`
- keep:
  - `station_key`
  - `station_version_key`
  - `city`
  - `station_id`
  - `station_name`
  - `capacity`
  - `latitude`
  - `longitude`
  - `valid_from_utc`
  - `valid_to_utc`
  - `is_current`

Trade-off:

- SCD2 is more complex than a latest-only station table.
- But it removes ambiguity around point-in-time station attributes and aligns with enterprise star-schema practice.
- Recommendation for Day 4: make `station_key` mandatory as the durable station business key and use `station_version_key` for fact-to-dimension joins.

### 7.2 Build `dim_time`

Purpose:

- centralize time-of-day fields used in features

Recommended logic:

- generate 5-minute buckets for a full day
- include:
  - `time_id`
  - `hour`
  - `minute`
  - `minute_of_day`
  - optional bucket label

Trade-off:

- You can compute `hour` directly in the feature table without a time dimension.
- But adding `dim_time` now makes the warehouse cleaner and keeps the star-schema direction intact.

Recommendation:

- create `dim_time` now, even if only `hour` is actively consumed today

### 7.3 Build `fct_station_status`

Purpose:

- create one normalized fact table from status staging, joined to dimensions and ready for feature engineering

Recommended columns:

- `city`
- `station_id`
- `station_key`
- `station_version_key`
- `snapshot_bucket_at_utc`
- `snapshot_bucket_at_paris`
- `date_id`
- `time_id`
- `bikes`
- `docks`
- `last_reported_at_utc`
- `last_reported_at_paris`

Important note:

Current station staging must not rely on `run_id + station_id` as analytical grain.
The base fact grain should be physically anchored on `city + snapshot_bucket_at_utc + station_id`.

Recommended Day 4 correction:

- keep `city` in all station joins
- add physical `snapshot_bucket_at_utc` and `snapshot_bucket_at_paris` to station staging first
- make station staging idempotent by `city + snapshot_bucket_at_utc`
- do not carry `capacity`, utilization metrics, or weather keys in the base fact
- join `dim_station` with as-of SCD2 semantics to get `station_version_key`

Strong recommendation:

- do not build formal feature models on top of `run_id` grain

---

## 8. Build the dbt intermediate layer next

### 8.1 Build `int_station_neighbors`

Purpose:

- replace Python BallTree neighbor computation with warehouse-owned neighbor relationships

Recommended logic:

- derive neighbor candidates from `dim_station`
- use Postgres SQL distance logic
- keep nearest K stations
- use fixed radius fallback

Recommended parameters for Day 4:

- `K = 5`
- `radius_km = 0.8`
- fallback to nearest available neighbors if none are inside radius

Trade-off:

- Python BallTree is more elegant and faster for large-scale nearest-neighbor problems
- dbt SQL is easier to version, test, and keep identical between training and inference

Recommendation:

- accept SQL simplicity over algorithmic elegance for this project scale

### 8.2 Build `int_station_status_enriched`

Purpose:

- create the one place where raw status is enriched with date, weather, station attributes, and lag context

Recommended responsibilities:

- join `fct_station_status` with `dim_station`
- join `dim_date`
- join weather using as-of semantics
- compute lag/lead-ready columns
- keep no final model-only aliases here unless they are reusable

Station enrichment rule:

- this is where station attributes such as `capacity`, `latitude`, and `longitude` should be exposed to feature models
- the base fact should remain lean and only carry station/time keys plus status measures

Trade-off:

- you could put all logic directly into the final feature model
- but that creates one very large SQL file that is hard to validate

Recommendation:

- keep enrichment in `intermediate`
- keep final ML contract naming in `features`

---

## 9. Build the dbt feature layer

This is the central Day 4 outcome.

### 9.1 Build `feat_station_snapshot_5min`

Purpose:

- one formal training row per station per 5-minute snapshot

Recommended responsibilities:

- derive:
  - `util_bikes`
  - `util_docks`
  - `delta_bikes_5m`
  - `delta_docks_5m`
  - rolling features
  - neighbor weighted features
  - `hour`
  - `dow`
  - `is_weekend`
  - `is_holiday`
  - weather features aligned to the ML subset of `dim_weather`
  - labels:
    - `target_bikes_t30`
    - `target_docks_t30`
    - `y_stockout_bikes_30`
    - `y_stockout_docks_30`

Required weather columns in the final training table:

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

Do not expose these as ML input columns in the final training table:

- `weather_main`
- `hourly_weather_main`
- `weather_description`
- `hourly_forecast_at`
- `source`
- `snapshot_bucket_at_utc`

Recommended grain:

- one row per `city + station_id + dt`

Recommended base columns to preserve:

- `city`
- `dt`
- `station_id`
- `capacity`
- `lat`
- `lon`
- `bikes`
- `docks`

These align directly with `REQUIRED_BASE` in `schema.py`.
They should come from the intermediate enrichment layer after joining the lean fact with SCD2 `dim_station`.

### 9.2 Build `feat_station_snapshot_latest`

Purpose:

- the single latest row per station for prediction

Recommended logic:

- select from `feat_station_snapshot_5min`
- keep only the latest `dt` per `city + station_id`
- exclude label fields if you want a cleaner serving table

Trade-off:

- you can expose all columns including labels and let Python ignore them
- or you can make a serving-clean table

Recommendation:

- keep the latest table serving-clean:
  - base columns
  - feature columns
  - optional metadata columns
  - no future labels

This keeps online prediction intent obvious.

Weather rule for the latest table:

- keep the same 11 weather features used by `feat_station_snapshot_5min`
- do not rename weather columns for online-only compatibility
- do not add `weather_main` or `hourly_weather_main` into serving inputs

### 9.3 Recommended Day 4 feature schema decision

Recommended final contract for Day 4:

- `feat_station_snapshot_5min`
  - includes required base columns, feature columns, and labels
- `feat_station_snapshot_latest`
  - includes required base columns and feature columns only

For weather specifically, both tables must expose the exact same 11 ML weather columns:

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

This is the cleanest split for training vs online serving.

---

## 10. Update dbt project configuration

Current [dbt_project.yml](/c:/Career/selfGrowth/projects/mlops-bikeshare-202508/dbt/bikeshare_dbt/dbt_project.yml)
defines only:

- `staging`
- `marts`

Day 4 should extend it to:

- `staging`
- `intermediate`
- `marts`
- `features`

Recommended materialization:

- `staging`: `view`
- `intermediate`: `view`
- `marts`: `table`
- `features`: `table`

Trade-off:

- `features` could be `view` during development for faster iteration
- but formal feature tables are easier to inspect, test, and serve as stable contracts when materialized as `table`

Recommendation:

- materialize final features as `table`

---

## 11. Add dbt tests before changing Python consumers

Do not change `train.py` first.
Make the feature contract testable first.

### 11.1 What to test in dbt

For `dim_station`:

- `station_version_key` unique
- `station_version_key` not null
- `station_key` not null
- exactly one `is_current = true` row per `station_key`
- no overlapping validity windows for the same `station_key`

For `fct_station_status`:

- grain key unique
- station relationship valid
- `snapshot_bucket_at_utc` not null
- `station_version_key` not null

For `feat_station_snapshot_5min`:

- `city`, `dt`, `station_id` not null
- revised feature columns expected by Python exist
- all 11 ML weather columns exist with the exact `dim_weather`-aligned names
- labels exist
- utilization fields within range

For `feat_station_snapshot_latest`:

- exactly one row per `city + station_id`
- no nulls in feature columns that are formally required
- the same 11 ML weather columns exist as in `feat_station_snapshot_5min`
- `weather_main` and `hourly_weather_main` are not present as serving feature columns

### 11.2 How to validate dbt after Day 4 models are added

Use this PowerShell order:

```powershell
dbt debug --project-dir dbt\bikeshare_dbt --profiles-dir dbt
dbt parse --project-dir dbt\bikeshare_dbt --profiles-dir dbt
dbt ls --project-dir dbt\bikeshare_dbt --profiles-dir dbt

dbt run --project-dir dbt\bikeshare_dbt --profiles-dir dbt --select dim_station dim_time fct_station_status
dbt test --project-dir dbt\bikeshare_dbt --profiles-dir dbt --select dim_station dim_time fct_station_status

dbt run --project-dir dbt\bikeshare_dbt --profiles-dir dbt --select int_station_neighbors int_station_status_enriched
dbt test --project-dir dbt\bikeshare_dbt --profiles-dir dbt --select int_station_neighbors int_station_status_enriched

dbt run --project-dir dbt\bikeshare_dbt --profiles-dir dbt --select feat_station_snapshot_5min feat_station_snapshot_latest
dbt test --project-dir dbt\bikeshare_dbt --profiles-dir dbt --select feat_station_snapshot_5min feat_station_snapshot_latest
```

Then inspect the outputs:

```powershell
docker compose exec dw-postgres psql -U velib -d velib_dw -c "SELECT COUNT(*) FROM analytics.dim_station;"
docker compose exec dw-postgres psql -U velib -d velib_dw -c "SELECT COUNT(*) FROM analytics.dim_time;"
docker compose exec dw-postgres psql -U velib -d velib_dw -c "SELECT COUNT(*) FROM analytics.fct_station_status;"
docker compose exec dw-postgres psql -U velib -d velib_dw -c "SELECT COUNT(*) FROM analytics.feat_station_snapshot_5min;"
docker compose exec dw-postgres psql -U velib -d velib_dw -c "SELECT COUNT(*) FROM analytics.feat_station_snapshot_latest;"
docker compose exec dw-postgres psql -U velib -d velib_dw -c "SELECT * FROM analytics.feat_station_snapshot_5min ORDER BY dt DESC LIMIT 5;"
docker compose exec dw-postgres psql -U velib -d velib_dw -c "SELECT * FROM analytics.feat_station_snapshot_latest LIMIT 5;"
```

---

## 12. Refactor `schema.py` only after dbt features exist

Current [schema.py](/c:/Career/selfGrowth/projects/mlops-bikeshare-202508/src/features/schema.py)
is still the Python-side contract.

Day 4 rule:

- keep `schema.py` as the Python consumer contract
- make the Python consumer contract follow the dbt / `dim_weather` weather contract
- update `schema.py` where necessary to reflect the new source-of-truth table structure

Recommended changes:

- revise the weather-related portion of `FEATURE_COLUMNS`
- keep `LABEL_COLUMNS`
- keep `REQUIRED_BASE`
- add a short note in comments that dbt feature tables are now the formal producer

Weather-related `FEATURE_COLUMNS` for Day 4 should be:

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

Trade-off:

- You could invert the ownership and make Python infer columns dynamically from SQL or YAML metadata.
- That is cleaner long-term, but overkill for today.

Recommendation:

- preserve `schema.py` as a stable explicit Python contract for now, but update its weather fields to the Day 4 contract

---

## 13. Refactor `train.py` from Athena to Postgres

Current [train.py](/c:/Career/selfGrowth/projects/mlops-bikeshare-202508/src/training/train.py)
still:

- imports `pyathena`
- reads `features_offline`
- splits by distinct Athena `dt`

Day 4 should remove that as the formal path.

### 13.1 Recommended replacement design

Replace Athena access with SQLAlchemy or psycopg2 against local Postgres.

Recommended source table:

- `analytics.feat_station_snapshot_5min`

Recommended local data-access helpers:

- `pg_engine(...)`
- `list_unique_dt_postgres(...)`
- `load_slice_postgres(...)`

Recommended behavior to preserve:

- same time-based split logic
- same anti-leakage gap logic
- same feature validation
- same MLflow logging

Weather contract behavior to change:

- training should validate the revised `FEATURE_COLUMNS`
- weather inputs should come from the 11 `dim_weather`-aligned ML weather columns
- training should not expect deprecated weather names or text weather fields

That means the refactor should change the data access layer, not the whole training pipeline behavior.

### 13.2 CLI design trade-off

Current CLI uses Athena-style args:

- `--database`
- `--workgroup`
- `--athena-output`
- `--region`

Recommended Day 4 trade-off:

- stop requiring Athena-only args
- add explicit Postgres args:
  - `--pg-host`
  - `--pg-port`
  - `--pg-db`
  - `--pg-user`
  - `--pg-password`
  - `--pg-schema`
- keep the old Athena args only if you still need legacy fallback

Recommendation:

- keep one optional legacy path only if absolutely necessary
- otherwise simplify the CLI and make Postgres the default and only formal mode

### 13.3 How to test training after the refactor

Syntax first:

```powershell
python -m py_compile src\training\train.py
```

Then run a small local window:

```powershell
python src\training\train.py `
  --city paris `
  --start "2026-03-01 00:00" `
  --end "2026-03-03 23:55" `
  --label y_stockout_bikes_30 `
  --model-type xgboost `
  --pg-host localhost `
  --pg-port 15432 `
  --pg-db velib_dw `
  --pg-user velib `
  --pg-password velib `
  --pg-schema analytics `
  --experiment bikeshare_local_day4
```

Then validate MLflow artifacts:

```powershell
Get-ChildItem mlruns
```

If you run MLflow UI locally:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## 14. Refactor online featurization to consume dbt latest features

Current [featurize_online.py](/c:/Career/selfGrowth/projects/mlops-bikeshare-202508/src/inference/featurize_online.py)
still:

- connects to Athena
- loads raw-ish views
- rebuilds weather alignment
- rebuilds neighbors
- calls the same Python `engineer(...)` logic

That is exactly what Day 4 is supposed to eliminate.

### 14.1 Recommended Day 4 replacement

`build_online_features(city)` should:

1. connect to Postgres
2. read from `analytics.feat_station_snapshot_latest`
3. validate that all `FEATURE_COLUMNS` exist
4. return:
   - `city`
   - `dt`
   - `station_id`
   - all `FEATURE_COLUMNS`

It should not:

- compute neighbors
- align weather
- engineer rolling windows
- perform feature logic

That work is already done in dbt.

Weather contract requirement:

- online serving must consume the same 11 weather columns used by offline training
- online serving must not depend on `weather_main` or `hourly_weather_main`
- online serving must not translate weather names back into legacy Python weather aliases

### 14.2 How to test online features after the refactor

Syntax:

```powershell
python -m py_compile src\inference\featurize_online.py
```

Run the script directly:

```powershell
python src\inference\featurize_online.py
```

If you want database inspection:

```powershell
docker compose exec dw-postgres psql -U velib -d velib_dw -c "SELECT dt, station_id FROM analytics.feat_station_snapshot_latest LIMIT 10;"
```

---

## 15. Refactor predictor to rely on the new online contract

Current [predictor.py](/c:/Career/selfGrowth/projects/mlops-bikeshare-202508/src/inference/predictor.py)
already calls `build_online_features(...)`, which is good.

That means Day 4 predictor refactor should stay narrow:

- keep predictor behavior mostly unchanged
- let the upstream feature source change from Python-engineered to dbt-produced

Recommended Day 4 rule:

- do not redesign output storage unless it blocks you
- only remove formal dependency on Athena-built features

Trade-off:

- You could also move prediction output from S3 to Postgres today.
- That is useful later, but not required to finish local-chain convergence.

Recommendation:

- Day 4 only guarantees input contract convergence
- leave prediction-output storage changes for the next stage unless they are trivial

### 15.1 How to test predictor after online feature refactor

Syntax:

```powershell
python -m py_compile src\inference\predictor.py
```

Run predictor with a local or staging endpoint if available:

```powershell
$env:SM_ENDPOINT = "bikeshare-prod"
python -m src.inference.predictor
```

If no endpoint is available yet, validate only the feature-loading half:

```powershell
python src\inference\featurize_online.py
```

That is still a valid Day 4 completion state.

---

## 16. Define the local MLflow Registry standard

Day 4 should not wait until AWS deployment to clean up model naming.

### 16.1 Recommended local Registry naming

Registry names:

- `paris_y_stockout_bikes_30_xgboost`
- `paris_y_stockout_docks_30_xgboost`
- `paris_y_stockout_bikes_30_lightgbm`
- `paris_y_stockout_docks_30_lightgbm`

Artifact path names inside a run:

- `base_model`
- `probability_model`
- `eval_summary`
- `feature_importance`

Recommended tags:

- `city=paris`
- `label=y_stockout_bikes_30`
- `model_type=xgboost`
- `feature_source=analytics.feat_station_snapshot_5min`
- `feature_contract=v1_dim_weather_aligned`

### 16.2 Why this matters now

If you keep generic names:

- later staging/prod promotion becomes ambiguous
- rollback targeting becomes harder
- dashboards and runbooks become unclear

This is a small Day 4 effort with high future payoff.

### 16.3 What to change in the helper scripts

In [register_model.py](/c:/Career/selfGrowth/projects/mlops-bikeshare-202508/pipelines/register_model.py):

- stop assuming a generic registry name
- register the explicit task-specific name
- update comments and examples

In [promote.py](/c:/Career/selfGrowth/projects/mlops-bikeshare-202508/pipelines/promote.py):

- keep the promotion helper
- update examples to use the new naming standard
- clarify that local Day 4 promotion is metadata-only, not endpoint deployment

### 16.4 Example local registration command

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

python pipelines\register_model.py `
  --run-id <RUN_ID> `
  --model-name paris_y_stockout_bikes_30_xgboost `
  --stage Staging
```

---

## 17. Practical Day 4 execution order

Use this order.
Do not jump around.

### 17.1 Stage A: verify Day 3 baseline

```powershell
. .\.venv\Scripts\Activate.ps1
docker compose up airflow-init
docker compose up -d airflow-postgres dw-postgres airflow-webserver airflow-scheduler
dbt debug --project-dir dbt\bikeshare_dbt --profiles-dir dbt
dbt parse --project-dir dbt\bikeshare_dbt --profiles-dir dbt
docker compose exec dw-postgres psql -U velib -d velib_dw -c "SELECT COUNT(*) FROM public.stg_station_status;"
docker compose exec dw-postgres psql -U velib -d velib_dw -c "SELECT COUNT(*) FROM analytics.dim_weather;"
docker compose exec dw-postgres psql -U velib -d velib_dw -c "SELECT COUNT(*) FROM analytics.dim_date;"
```

### 17.2 Stage B: add marts and intermediate models

```powershell
dbt run --project-dir dbt\bikeshare_dbt --profiles-dir dbt --select dim_station dim_time fct_station_status int_station_neighbors int_station_status_enriched
dbt test --project-dir dbt\bikeshare_dbt --profiles-dir dbt --select dim_station dim_time fct_station_status int_station_neighbors int_station_status_enriched
```

### 17.3 Stage C: add final feature models

```powershell
dbt run --project-dir dbt\bikeshare_dbt --profiles-dir dbt --select feat_station_snapshot_5min feat_station_snapshot_latest
dbt test --project-dir dbt\bikeshare_dbt --profiles-dir dbt --select feat_station_snapshot_5min feat_station_snapshot_latest

docker compose exec dw-postgres psql -U velib -d velib_dw -c "SELECT COUNT(*) FROM analytics.feat_station_snapshot_5min;"
docker compose exec dw-postgres psql -U velib -d velib_dw -c "SELECT COUNT(*) FROM analytics.feat_station_snapshot_latest;"
```

### 17.4 Stage D: refactor Python consumers

```powershell
python -m py_compile src\features\schema.py
python -m py_compile src\training\train.py
python -m py_compile src\inference\featurize_online.py
python -m py_compile src\inference\predictor.py
python -m py_compile pipelines\register_model.py
python -m py_compile pipelines\promote.py
```

### 17.5 Stage E: validate local training

```powershell
python src\training\train.py `
  --city paris `
  --start "2026-03-01 00:00" `
  --end "2026-03-03 23:55" `
  --label y_stockout_bikes_30 `
  --model-type xgboost `
  --pg-host localhost `
  --pg-port 15432 `
  --pg-db velib_dw `
  --pg-user velib `
  --pg-password velib `
  --pg-schema analytics `
  --experiment bikeshare_local_day4
```

### 17.6 Stage F: validate online contract

```powershell
python src\inference\featurize_online.py
```

### 17.7 Stage G: register the model locally

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

python pipelines\register_model.py `
  --run-id <RUN_ID> `
  --model-name paris_y_stockout_bikes_30_xgboost `
  --stage Staging
```

---

## 18. If a stage fails, reset only that stage

### Reset dbt marts and features only

```powershell
docker compose exec dw-postgres psql -U velib -d velib_dw -c "DROP SCHEMA IF EXISTS analytics CASCADE; CREATE SCHEMA analytics;"
dbt run --project-dir dbt\bikeshare_dbt --profiles-dir dbt
dbt test --project-dir dbt\bikeshare_dbt --profiles-dir dbt
```

### Reset MLflow local metadata only if necessary

Use caution here because this can remove local experiment history.

```powershell
Get-ChildItem mlruns
```

If you intentionally want a clean local run history, handle that separately and only after you confirm you do not need prior run IDs.

### Re-run only training

```powershell
python src\training\train.py `
  --city paris `
  --start "2026-03-01 00:00" `
  --end "2026-03-03 23:55" `
  --label y_stockout_bikes_30 `
  --model-type xgboost `
  --pg-host localhost `
  --pg-port 15432 `
  --pg-db velib_dw `
  --pg-user velib `
  --pg-password velib `
  --pg-schema analytics `
  --experiment bikeshare_local_day4
```

### Re-run only online feature validation

```powershell
python src\inference\featurize_online.py
```

---

## 19. Common issues

### dbt feature models fail because the station grain is unstable

Check whether `stg_station_status` still lacks:

- `city`
- `snapshot_bucket_at_utc`

If so, fix staging grain before trusting the feature layer.

### Training works but feature validation fails

Check:

- dbt output column names exactly match revised `FEATURE_COLUMNS`
- required base columns still match `REQUIRED_BASE`
- the weather portion of `FEATURE_COLUMNS` is exactly:
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

### Online features return zero rows

Check:

- `analytics.feat_station_snapshot_latest` has rows
- latest-row logic is partitioned by both `city` and `station_id`
- `dt` format still matches `"YYYY-MM-DD-HH-mm"`
- online weather columns still match the same 11-column contract used by training

### MLflow registration script fails

Check:

- `MLFLOW_TRACKING_URI` is set
- the run actually logged the expected model artifact path
- registry name is spelled exactly as intended

### Predictor still indirectly depends on Athena

Check:

- `featurize_online.py` no longer imports Athena helpers
- `predictor.py` no longer assumes Athena-built features

---

## 20. Day 4 checklist

- dbt project includes `intermediate` and `features`
- `dim_station` builds
- `dim_time` builds
- `fct_station_status` builds
- `int_station_neighbors` builds
- `int_station_status_enriched` builds
- `feat_station_snapshot_5min` builds
- `feat_station_snapshot_latest` builds
- dbt tests for feature tables pass
- `train.py` reads Postgres feature tables
- local training run completes
- `featurize_online.py` reads `feat_station_snapshot_latest`
- predictor input contract matches dbt feature output
- MLflow naming standard is documented and used
- no formal training or online feature path still depends on Athena

If all of these are done, the local chain is converged and ready for the next phase:

- always-on runtime
- dashboard migration
- AWS staging/prod prediction deployment
- admission gate and rollback orchestration
