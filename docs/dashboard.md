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
- current target
- endpoint name
- model version
- decision threshold

### Required Sections

1. Map of latest predictions
2. Top-N risk table
3. Station prediction history
4. Model health
5. System health
6. Data freshness

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

- Page config: "Velib Paris — Station Risk Monitor", wide layout
- Reads all configuration from `.streamlit/secrets.toml`
- Builds connections: PostgreSQL (SQLAlchemy), S3 (boto3), CloudWatch (boto3)
- Sidebar: target selector (bikes/docks), top-N slider, history limit slider
- Renders status cards, alert banner, and five tabs

### `app/dashboard/s3_loader.py`

| Function | Description |
|---|---|
| `load_latest_predictions(s3, bucket, target, city)` | Reads the latest S3 Parquet prediction file for the selected target |
| `load_prediction_history(s3, bucket, target, city, n)` | Reads the last N prediction files per station |
| `load_quality_recent(s3, bucket, target, city)` | Reads quality metrics from the last 24 hours |

All functions use target-aware S3 prefixes from `src.config.naming`. Return format: `DataFrame[station_id, ts, score]`.

### `app/dashboard/queries.py`

| Function | Description |
|---|---|
| `load_station_info(engine, city)` | Queries `analytics.feat_station_snapshot_latest` for station metadata (lat, lon, name, capacity) |
| `load_freshness(engine)` | Checks the latest `dt` per staging and feature table to determine freshness status |

SQL identifiers are validated before use. See `app/dashboard/utils.py`.

### `app/dashboard/views.py`

| Function | Description |
|---|---|
| `render_status_cards(target, env, model_version, threshold)` | Environment badge + 4 metric cards at the top |
| `render_alert_banner(df, threshold)` | High/medium/low risk summary banner |
| `render_prediction_map(df, station_info)` | Folium map with station markers colored by risk score |
| `render_top_risk_table(df, station_info, n)` | Sorted risk ranking table |
| `render_history_chart(df, station_id)` | Plotly time-series score chart for a single station |
| `render_metric_section(cw, endpoint, target, env, city)` | CloudWatch metrics display |
| `render_freshness_table(engine)` | Data freshness status per source table |

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
