# Day 1 Guide (Beginner-Friendly): Project Setup & Planning for Vélib’ + AWS Free Tier

This guide walks you through **exactly** what to do in your first ~5 hours.

---

## 0) What you will finish by end of Day 1

By the end of this session, you should have:

1. Cloned and explored your current repository structure.
2. Set up Python 3.11 and a virtual environment.
3. Installed dependencies.
4. Identified and tested the two Vélib’ GBFS real-time endpoints.
5. Confirmed GBFS + weather sources and designed a snapshot-based ML dataset (no trip history required).
6. Designed the star schema for 30-minute bike-shortage prediction.
7. Started a PostgreSQL data warehouse environment (Docker on local machine) and captured connection settings for your project.
8. (Optional AWS path) Prepared an AWS RDS/PostgreSQL alternative suitable for free-tier style learning.

---

## 1) Review existing repository (30–45 min)

### Step 1.1: Clone and open project

```bash
git clone <your-repo-url>
cd mlops-bikeshare
```

### Step 1.2: Explore top-level folders you mentioned

```bash
ls
```

Focus on:

- `app/`: dashboard or app entrypoints.
- `src/`: ingestion, feature engineering, training, inference code.
- `docs/`: architecture and runbooks.
- `pipelines/`: deployment/training pipeline scripts.
- `.github/workflows/`: CI/CD automation.

### Step 1.3: Read README first (don’t skip)

```bash
sed -n '1,220p' README.md
```

Write quick notes in a file `docs/day1_notes.md`:

- What problem this repo solves.
- What data sources are currently used.
- How training/inference currently runs.
- Any commands for local development.

### Step 1.4: Quick architecture scan

Read these docs if available:

- `docs/architecture.md`
- `docs/data_contract.md`
- `docs/training_eval.md`
- `docs/cicd.md`

Do not deep-dive yet; only build a mental model.

---

## 2) Set up local Python environment (45–60 min)

> Your requested command sequence is correct. Use exactly this flow.

### Step 2.1: Confirm Python 3.11 exists

```bash
python3 --version
```

If output is not 3.11, install Python 3.11 first (OS-specific package manager).

### Step 2.2: Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should now see `(.venv)` in your shell prompt.

### Step 2.3: Upgrade packaging tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

### Step 2.4: Install project dependencies

```bash
pip install -r requirements.txt
```

If you also do development/testing tasks:

```bash
pip install -r requirements-dev.txt
```

### Step 2.5: Validate imports quickly

```bash
python - <<'PY'
import pandas, requests
print('Environment looks good')
PY
```

---

## 3) Research Vélib’ real-time GBFS data (45–60 min)

You already have the correct base URL:

- `https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/`


### Step 3.0: Understand `system_information` and `gbfs` first (important)

Before using station endpoints, query these two metadata feeds.

**macOS/Linux (bash):**

```bash
curl -s "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/system_information.json" | head -c 500
curl -s "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/gbfs.json" | head -c 500
```

**Windows PowerShell (recommended):**

```powershell
$sys = Invoke-RestMethod -Uri "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/system_information.json"
$gbfs = Invoke-RestMethod -Uri "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/gbfs.json"
$sys | ConvertTo-Json -Depth 6
$gbfs | ConvertTo-Json -Depth 6
```

> PowerShell note: `curl` is an alias of `Invoke-WebRequest`, and `head` is not a native PowerShell command. Use `Invoke-RestMethod` / `Invoke-WebRequest` instead.

- `system_information` tells you **who/where/timezone/language** for the bike system.
  - Use it to set timezone assumptions and project metadata correctly.
- `gbfs.json` is a **feed catalog** (directory of all available GBFS endpoints).
  - Use it so your code can dynamically discover URLs (`station_status`, `station_information`, etc.) instead of hardcoding them.

### Step 3.1: Check station metadata endpoint

```bash
# macOS/Linux
curl -s "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_information.json" | head -c 600
```

```powershell
# Windows PowerShell
(Invoke-RestMethod -Uri "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_information.json") | ConvertTo-Json -Depth 8
```

You should see JSON with station-level static fields (IDs, names, coordinates, capacity).

### Step 3.2: Check station status endpoint

```bash
# macOS/Linux
curl -s "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_status.json" | head -c 600
```

```powershell
# Windows PowerShell
(Invoke-RestMethod -Uri "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_status.json") | ConvertTo-Json -Depth 8
```

You should see JSON with live occupancy/availability fields.

### Step 3.3: Confirm no auth requirement

Both endpoints should return data directly via HTTPS, with no token/auth header.

### Step 3.4: Save one sample from each for reproducibility

```bash
# macOS/Linux
mkdir -p data/samples/velib
curl -s "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_information.json" -o data/samples/velib/station_information.sample.json
curl -s "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_status.json" -o data/samples/velib/station_status.sample.json
```

```powershell
# Windows PowerShell
New-Item -ItemType Directory -Force -Path "data/samples/velib" | Out-Null
Invoke-WebRequest -Uri "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_information.json" -OutFile "data/samples/velib/station_information.sample.json"
Invoke-WebRequest -Uri "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_status.json" -OutFile "data/samples/velib/station_status.sample.json"
```

### Step 3.5: Inspect sample schema with Python

```bash
python - <<'PY'
import json
for fn in [
    'data/samples/velib/station_information.sample.json',
    'data/samples/velib/station_status.sample.json'
]:
    with open(fn) as f:
        d = json.load(f)
    print('\n===', fn, '===')
    print('Top-level keys:', list(d.keys()))
    stations = d.get('data', {}).get('stations', [])
    print('Rows:', len(stations))
    if stations:
        print('Sample station keys:', sorted(stations[0].keys())[:20])
PY
```

Record key fields into `docs/day1_notes.md`.

---

## 4) Check weather ingestion pipeline (60–75 min)

Before modeling, make sure weather ingestion works end-to-end because your feature set depends on it.

### Step 4.1: Confirm weather provider mode and credentials

In this repo, weather ingest supports Meteostat (`official` or `rapidapi`) with Open-Meteo fallback.

Set env vars (PowerShell example):

```powershell
$env:BUCKET = "<your-s3-bucket>"
$env:CITY = "paris"
$env:METEOSTAT_PROVIDER = "rapidapi"
$env:METEOSTAT_API_KEY = "<your-rapidapi-key>"
$env:METEOSTAT_ALT = "35"
```

### Step 4.2: Validate RapidAPI call manually (connectivity test)

Your command style is correct. For a quick API connectivity check in PowerShell:

```powershell
$headers = @{}
$headers.Add("x-rapidapi-key", "<your-rapidapi-key>")
$headers.Add("x-rapidapi-host", "meteostat.p.rapidapi.com")
$response = Invoke-WebRequest -Uri 'https://meteostat.p.rapidapi.com/point/monthly?lat=48.8566&lon=2.3522&alt=35&start=2020-01-01&end=2020-12-31' -Method GET -Headers $headers
$response.StatusCode
```

If `StatusCode = 200`, API/auth/network are OK.

### Step 4.3: Run repo weather ingestion locally

```bash
# from repo root
python -m src.ingest.weather_ingest
```

Expected behavior:

- It fetches Meteostat data for `CITY=paris` (or falls back to Open-Meteo if Meteostat fails).
- It validates and normalizes rows.
- It writes gzip JSON to S3 key pattern:
  - `raw/weather_hourly/city=paris/dt=YYYY-MM-DD-HH-MM/data.json.gz`

### Step 4.4: Verify data landed in S3

```bash
aws s3 ls s3://<your-s3-bucket>/raw/weather_hourly/city=paris/ --recursive | tail -n 5
```

### Step 4.5: If weather ingest fails, check these first

1. `METEOSTAT_API_KEY` missing or invalid.
2. Wrong provider mode (`METEOSTAT_PROVIDER=official` without official API key).
3. `CITY` not configured in `CITY_COORDS` in `src/ingest/weather_ingest.py`.
4. S3 bucket permissions (`s3:PutObject`) missing for runtime identity.

---

## 5) Define and document the star schema for snapshot forecasting (45–60 min)

Use this schema for your current objective (no trip table needed now).

### Step 5.1: Logical model

- **dim_station**
  - `station_id` (PK)
  - `name`
  - `latitude`
  - `longitude`
  - `capacity`

- **dim_date**
  - `date_id` (PK)
  - `date`
  - `day_of_week`
  - `month`
  - `year`

- **dim_time**
  - `time_id` (PK)
  - `hour`
  - `minute`
  - `is_peak_hour`

- **dim_weather**
  - `weather_id` (PK)
  - `observed_at`
  - `temperature_c`
  - `precipitation_mm`
  - `wind_speed_kmh`
  - `humidity_pct`

- **fact_station_status**
  - `station_status_id` (PK)
  - `observed_at`
  - `station_id` (FK)
  - `date_id` (FK)
  - `time_id` (FK)
  - `weather_id` (FK)
  - `num_bikes_available`
  - `num_docks_available`
  - `is_renting`
  - `is_returning`
  - `bike_shortage_30m` (label)

### Step 5.2: Draw diagram

Options:

- draw.io (fast and free)
- dbdiagram.io
- even hand-drawn photo in docs for now

Save artifact as: `docs/diagrams/day1_star_schema.png` (or `.drawio`).

### Step 5.3: Add SQL DDL skeleton

Use `docs/sql/day1_star_schema.sql` (already prepared) as your starter schema.

---

## 6) Prepare warehouse environment (AWS free-tier friendly path) (60–75 min)

You asked for AWS free tier, so here are **two practical options**:

## Option A (recommended for Day 1): Local PostgreSQL via Docker

This is fastest and closest to your given instruction.

### Step A1: Run PostgreSQL 15 container

```bash
docker run --name velib-dw \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=velib \
  -e POSTGRES_USER=velib \
  -e POSTGRES_DB=velib_dw \
  -d postgres:15
```

### Step A2: Verify container health

```bash
docker ps
```

### Step A3: Connect with psql

```bash
psql "postgresql://velib:velib@localhost:5432/velib_dw" -c "SELECT version();"
```

### Step A4: Apply schema DDL

```bash
psql "postgresql://velib:velib@localhost:5432/velib_dw" -f docs/sql/day1_star_schema.sql
```

### Step A5: Save connection settings in `.env`

```env
DW_HOST=localhost
DW_PORT=5432
DW_DB=velib_dw
DW_USER=velib
DW_PASSWORD=velib
```

> Add `.env` to `.gitignore` if not already ignored.

---

## Option B (AWS route): Amazon RDS PostgreSQL (free-tier style)

> **Important:** Redshift is usually **not** ideal for strict free-tier beginners due to cost risk. Start with RDS PostgreSQL first.

### Step B1: Create RDS PostgreSQL instance

In AWS Console:

1. Go to **RDS → Databases → Create database**.
2. Choose **Standard create**.
3. Engine: **PostgreSQL**.
4. Template: **Free tier** (if shown in your account/region).
5. DB instance identifier: `velib-dw-dev`.
6. Master username/password: store securely.
7. Storage: minimal default (monitor costs).
8. Public access: **Yes** for learning (lock down by SG IP).
9. VPC security group: allow inbound TCP 5432 from **your IP only**.
10. Create DB.

### Step B2: Wait until status is “Available” and copy endpoint

Example endpoint:

`velib-dw-dev.abc123xyz.us-east-1.rds.amazonaws.com`

### Step B3: Connect from local terminal

```bash
psql "postgresql://<user>:<password>@<rds-endpoint>:5432/postgres" -c "SELECT current_database();"
```

### Step B4: Create target database and tables

```bash
psql "postgresql://<user>:<password>@<rds-endpoint>:5432/postgres" -c "CREATE DATABASE velib_dw;"
psql "postgresql://<user>:<password>@<rds-endpoint>:5432/velib_dw" -f docs/sql/day1_star_schema.sql
```

### Step B5: Store credentials in `.env`

```env
DW_HOST=<rds-endpoint>
DW_PORT=5432
DW_DB=velib_dw
DW_USER=<user>
DW_PASSWORD=<password>
```

---

## 7) Document Airflow/connection info now (15 min)

Even if Airflow setup is later, prepare a connection spec today.

In `docs/day1_notes.md`, include:

- Connection ID suggestion: `velib_dw`
- Conn type: `Postgres`
- Host, schema (DB), login, password, port
- SSL mode (if RDS, usually `require`)

If using Airflow CLI later, this is typical:

```bash
airflow connections add 'velib_dw' \
  --conn-type 'postgres' \
  --conn-host "$DW_HOST" \
  --conn-schema "$DW_DB" \
  --conn-login "$DW_USER" \
  --conn-password "$DW_PASSWORD" \
  --conn-port "$DW_PORT"
```

---

## 8) Day-1 deliverables checklist (copy/paste and tick)

- [ ] Repo cloned and README reviewed.
- [ ] `.venv` created with Python 3.11.
- [ ] Dependencies installed successfully.
- [ ] GBFS endpoints validated (`station_information`, `station_status`).
- [ ] Sample GBFS JSON saved in `data/samples/velib/`.
- [ ] Weather ingestion run once and S3 output verified.
- [ ] Snapshot label definition decided (`bike_shortage_30m`).
- [ ] Feature mapping drafted in `docs/day1_notes.md`.
- [ ] Star schema diagram created and saved in `docs/diagrams/`.
- [ ] SQL DDL file created (`docs/sql/day1_star_schema.sql`).
- [ ] PostgreSQL warehouse running (local Docker or RDS).
- [ ] `.env` or connection doc prepared.

---

## 9) Suggested plan for Day 2 (so you have continuity)

1. Build ingestion script for GBFS snapshots (`src/ingest/gbfs_ingest.py` extension).
2. Build weather + station-status join at 5-minute granularity.
3. Generate `bike_shortage_30m` labels with a 30-minute forward window.
4. Add simple quality checks (nulls, FK consistency, label balance).
5. Automate with a basic pipeline (cron/Airflow/GitHub Actions).

---

## Beginner tips to avoid common mistakes

- Always activate `.venv` before running scripts.
- Keep secrets in `.env`, never commit credentials.
- Start small: one day of data, then scale.
- Validate schema early; mismatched station IDs are common.
- Track costs in AWS Billing dashboard from Day 1.

