# Dashboard

## Purpose

The Streamlit dashboard serves both business users and operators. It shows real-time stockout risk predictions for Paris Vélib stations and exposes model health and data freshness signals.

---

## Specification

### Required Scope Selector

- Target selector with:
  - `Bike stockout`
  - `Dock stockout`
- City selector may remain fixed to `paris` for this project
- Environment indicator must be visible

### Required Status Cards

The top section must show:
- pipeline state
- current target
- endpoint name
- model version
- latest prediction timestamp

### Required Sections

1. Map of latest predictions
2. Top-N risk table
3. Station prediction history
4. Model health
5. System health
6. Data status / freshness

### Target-Aware Requirements

- No chart may hardcode `yhat_bikes` or `y_stockout_bikes_30`
- Endpoint names must resolve from target plus environment
- CloudWatch queries must include `TargetName`
- Prediction and quality prefixes must include `target=`
- Switching target must update all cards, charts, and prefixes together

### Production Restrictions

- No debug-only publish controls in formal mode
- No target inference based only on endpoint name
- No legacy Athena `features_offline` dependency in formal dashboard paths

### Validation Checklist

- bikes view shows bikes label/score columns
- docks view shows docks label/score columns
- prod/staging endpoint names are correct
- model health charts use the selected target dimensions
- non-technical users can tell which target they are viewing within five seconds

---

## Current Implementation

The dashboard is implemented in `app/dashboard/` with the following modules:

### Entry Point: `app/dashboard.py`

- Page config: "Velib Paris Station Risk Monitor", wide layout
- Reads all configuration from `.streamlit/secrets.toml`
- Builds connections: PostgreSQL (SQLAlchemy), S3 (boto3), CloudWatch (boto3)
- Sidebar: target selector (bikes/docks), top-N slider, history limit slider
- Renders status cards, alert banner, and five tabs: `Live Ops`, `Station History`, `Prediction Quality`, `System Health`, `Data Status`

### `app/dashboard/s3_loader.py`

| Function | Description |
|---|---|
| `load_latest_predictions(bucket, city, target_name, s3_client)` | Reads the latest S3 Parquet prediction artifact for the selected target |
| `load_prediction_history(bucket, city, target_name, station_id, n_periods, s3_client)` | Reads the last N prediction artifacts for one selected station |
| `load_latest_quality_status(bucket, city, target_name, s3_client)` | Reads the latest mature quality artifact and validates target-aware label/score columns |

All functions use target-aware S3 prefixes from `src.config.naming` and return `ArtifactLoadResult` wrappers instead of raw DataFrames.

### `app/dashboard/queries.py`

| Function | Description |
|---|---|
| `load_station_info(engine, schema, city)` | Queries `analytics.feat_station_snapshot_latest` plus `dim_station` for station metadata and latest serving context |
| `load_freshness(engine, schema, city, tables)` | Checks the latest `dt` per monitored feature table and returns a `FreshnessLoadResult` |

SQL identifiers are validated before use. See `app/dashboard/utils.py`.

### `app/dashboard/views.py`

| Function | Description |
|---|---|
| `render_status_cards(...)` | Environment badge, pipeline state, target, endpoint, model version, and latest prediction timestamp |
| `render_alert_banner(...)` | Stale-artifact warning or current high/medium/low risk summary |
| `render_prediction_map(...)` | Folium map with station markers colored by current risk score |
| `render_top_risk_table(...)` | Sorted risk ranking table for the selected target |
| `render_history_chart(...)` | Plotly time-series score chart for the selected station |
| `render_metric_section(...)` | CloudWatch metric cards plus sparkline charts |
| `render_data_status_table(...)` | Source freshness plus feature/prediction/quality status using schedule-aware expected lag and missed-cycle severity |

Color palette: `#e63946` (red = high risk), `#f4a261` (orange = medium), `#2a9d8f` (teal = fresh/ok).

### `app/dashboard/utils.py`

- `validate_pg_identifier(name)`: SQL injection prevention — allows only alphanumeric characters and underscores. Used before any identifier is passed to a query.

### `app/dashboard/targeting.py`

- Resolves the active target name, endpoint name, and CloudWatch dimension values based on the sidebar selection.

---

## Configuration

### `.streamlit/secrets.toml` — Required Fields

```toml
[aws]
profile = "your-aws-profile"
region = "eu-west-3"
bucket = "bikeshare-paris-387706002632-eu-west-3"

[app]
city = "paris"
environment = "staging"         # or "production"
model_version = "v1.0"
threshold = 0.5
```

### PostgreSQL Connection

The dashboard reads `PG*` environment variables or Streamlit secrets for the database connection:

```
PGHOST=localhost
PGPORT=15432
PGDATABASE=velib_dw
PGUSER=velib
PGPASSWORD=velib
```

---

## Local Run

```powershell
# Activate virtual environment
.\.venv\Scripts\activate

# Install dashboard dependencies
pip install -r requirements-app.txt

# Set PostgreSQL variables
$env:PGHOST = "localhost"
$env:PGPORT = "15432"
$env:PGDATABASE = "velib_dw"
$env:PGUSER = "velib"
$env:PGPASSWORD = "velib"

# Run the dashboard
streamlit run app/dashboard.py
```

Or via Docker Compose (EC2 runtime):

```bash
docker compose up -d dashboard
# Access at http://localhost:8501
```

---

## Dashboard Dependencies

Tracked separately in `requirements-app.txt`:

```
streamlit>=1.35.0
streamlit-folium>=0.22.0
plotly>=5.20.0
folium>=0.17.0
```

These are not included in the main `requirements.txt` to keep the inference/training environment lean.
