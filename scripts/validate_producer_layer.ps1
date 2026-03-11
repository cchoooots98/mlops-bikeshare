<#
  Validate whether the dbt producer layer is stable enough for handoff to Python consumers.

  What this script checks:
  1) dbt connectivity and parseability
  2) key dbt producer tests for marts/intermediate/features
  3) actual relations and row counts in analytics schema
  4) obvious documentation drift that still claims feature layers are only "planned"

  Exit code:
  - 0: all checks passed
  - 1: one or more checks failed

  Usage:
    powershell -ExecutionPolicy Bypass -File .\scripts\validate_producer_layer.ps1
#>

param(
  [string]$ProjectDir = "dbt\bikeshare_dbt",
  [string]$ProfilesDir = "dbt",
  [string]$PgHost = "localhost",
  [int]$PgPort = 15432,
  [string]$PgDb = "velib_dw",
  [string]$PgUser = "velib",
  [string]$PgPassword = "velib"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$DbtExe = Join-Path $RepoRoot ".venv\Scripts\dbt.exe"
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $DbtExe)) {
  Write-Host "FAIL: dbt executable not found at $DbtExe"
  exit 1
}

if (-not (Test-Path $PythonExe)) {
  Write-Host "FAIL: python executable not found at $PythonExe"
  exit 1
}

$script:Failures = New-Object System.Collections.Generic.List[string]
$script:Warnings = New-Object System.Collections.Generic.List[string]

function Run-Step {
  param(
    [string]$Name,
    [scriptblock]$Action
  )

  Write-Host ""
  Write-Host "=== $Name ==="

  try {
    & $Action
    Write-Host "PASS: $Name"
  }
  catch {
    $message = $_.Exception.Message
    Write-Host "FAIL: $Name"
    Write-Host $message
    $script:Failures.Add("${Name}: ${message}") | Out-Null
  }
}

function Add-Warning {
  param([string]$Message)
  Write-Host "WARN: $Message"
  $script:Warnings.Add($Message) | Out-Null
}

Run-Step "dbt debug" {
  & $DbtExe debug --project-dir $ProjectDir --profiles-dir $ProfilesDir
  if ($LASTEXITCODE -ne 0) {
    throw "dbt debug returned exit code $LASTEXITCODE"
  }
}

Run-Step "dbt parse" {
  & $DbtExe parse --project-dir $ProjectDir --profiles-dir $ProfilesDir
  if ($LASTEXITCODE -ne 0) {
    throw "dbt parse returned exit code $LASTEXITCODE"
  }
}

Run-Step "dbt ls" {
  & $DbtExe ls --project-dir $ProjectDir --profiles-dir $ProfilesDir
  if ($LASTEXITCODE -ne 0) {
    throw "dbt ls returned exit code $LASTEXITCODE"
  }
}

Run-Step "dbt producer tests" {
  $output = & $DbtExe test `
    --project-dir $ProjectDir `
    --profiles-dir $ProfilesDir `
    --select dim_station dim_time dim_weather fct_station_status int_station_neighbors int_station_status_enriched feat_station_snapshot_5min feat_station_snapshot_latest `
    2>&1

  $output | ForEach-Object { Write-Host $_ }

  if ($LASTEXITCODE -ne 0) {
    if ($output -match "WinError 5" -or $output -match "Access is denied") {
      throw "dbt test failed due to local Windows permission issue. Rerun this script in an elevated PowerShell first. If it still fails, run the same dbt test command inside a clean shell after closing IDE-integrated terminals."
    }
    throw "dbt test returned exit code $LASTEXITCODE"
  }
}

Run-Step "analytics relation and row-count smoke check" {
  $env:PGHOST = $PgHost
  $env:PGPORT = [string]$PgPort
  $env:PGDATABASE = $PgDb
  $env:PGUSER = $PgUser
  $env:PGPASSWORD = $PgPassword

  @'
import os
import psycopg2

required_relations = {
    "table": [
        "dim_station",
        "dim_time",
        "dim_weather",
        "fct_station_status",
        "int_station_neighbors",
        "int_station_status_enriched",
        "feat_station_snapshot_5min",
    ],
    "view": [
        "feat_station_snapshot_latest",
    ],
}

count_queries = [
    "select count(*) from analytics.dim_station",
    "select count(*) from analytics.dim_time",
    "select count(*) from analytics.dim_weather",
    "select count(*) from analytics.fct_station_status",
    "select count(*) from analytics.int_station_neighbors",
    "select count(*) from analytics.int_station_status_enriched",
    "select count(*) from analytics.feat_station_snapshot_5min",
    "select count(*) from analytics.feat_station_snapshot_latest",
]

feature_columns_5min = {
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
    "has_neighbors_within_radius",
    "neighbor_count_within_radius",
    "target_bikes_t30",
    "target_docks_t30",
    "y_stockout_bikes_30",
    "y_stockout_docks_30",
}

feature_columns_latest = {
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
    "has_neighbors_within_radius",
    "neighbor_count_within_radius",
}

forbidden_latest = {
    "target_bikes_t30",
    "target_docks_t30",
    "y_stockout_bikes_30",
    "y_stockout_docks_30",
}

conn = psycopg2.connect(
    host=os.environ["PGHOST"],
    port=os.environ["PGPORT"],
    dbname=os.environ["PGDATABASE"],
    user=os.environ["PGUSER"],
    password=os.environ["PGPASSWORD"],
)
cur = conn.cursor()

cur.execute("select tablename from pg_tables where schemaname = 'analytics'")
tables = {row[0] for row in cur.fetchall()}
cur.execute("select viewname from pg_views where schemaname = 'analytics'")
views = {row[0] for row in cur.fetchall()}

missing_tables = sorted(set(required_relations["table"]) - tables)
missing_views = sorted(set(required_relations["view"]) - views)

if missing_tables:
    raise SystemExit(f"missing analytics tables: {missing_tables}")
if missing_views:
    raise SystemExit(f"missing analytics views: {missing_views}")

for sql in count_queries:
    cur.execute(sql)
    count = cur.fetchone()[0]
    print(sql, "=>", count)
    if count <= 0:
        raise SystemExit(f"empty relation detected for query: {sql}")

def column_set(name):
    cur.execute(
        """
        select column_name
        from information_schema.columns
        where table_schema = 'analytics'
          and table_name = %s
        order by ordinal_position
        """,
        (name,),
    )
    return {row[0] for row in cur.fetchall()}

cols_5min = column_set("feat_station_snapshot_5min")
cols_latest = column_set("feat_station_snapshot_latest")

missing_5min = sorted(feature_columns_5min - cols_5min)
missing_latest = sorted(feature_columns_latest - cols_latest)
present_forbidden_latest = sorted(forbidden_latest & cols_latest)

if missing_5min:
    raise SystemExit(f"feat_station_snapshot_5min missing required columns: {missing_5min}")
if missing_latest:
    raise SystemExit(f"feat_station_snapshot_latest missing required columns: {missing_latest}")
if present_forbidden_latest:
    raise SystemExit(f"feat_station_snapshot_latest still exposes label columns: {present_forbidden_latest}")

cur.close()
conn.close()
'@ | & $PythonExe -

  if ($LASTEXITCODE -ne 0) {
    throw "analytics smoke check returned exit code $LASTEXITCODE"
  }
}

Run-Step "documentation drift scan" {
  $driftRules = @(
    @{ Path = "docs\data_model.md"; Pattern = "not implemented in this phase|Planned dbt Layers" },
    @{ Path = "docs\feature_store.md"; Pattern = "Planned next flow|planned feature layer|Ownership target: dbt will become" },
    @{ Path = "docs\training_eval.md"; Pattern = "still reads `features_offline`|planned but not implemented" }
  )

  foreach ($rule in $driftRules) {
    if (-not (Test-Path $rule.Path)) {
      Add-Warning "Missing expected documentation file: $($rule.Path)"
      continue
    }

    $matches = Select-String -Path $rule.Path -Pattern $rule.Pattern -AllMatches
    if ($matches) {
      foreach ($match in $matches) {
        Add-Warning "Documentation drift in $($rule.Path): line $($match.LineNumber): $($match.Line.Trim())"
      }
    }
  }
}

Write-Host ""
Write-Host "=== Summary ==="

if ($Warnings.Count -gt 0) {
  Write-Host "Warnings:"
  foreach ($warning in $Warnings) {
    Write-Host " - $warning"
  }
}

if ($Failures.Count -gt 0) {
  Write-Host "Failures:"
  foreach ($failure in $Failures) {
    Write-Host " - $failure"
  }
  exit 1
}

Write-Host "Producer layer validation passed."
if ($Warnings.Count -gt 0) {
  Write-Host "Producer layer is executable, but documentation still needs cleanup."
}
exit 0
