# EC2 Release Update Guide

## Purpose

Use this guide after you have already:

- validated the change locally
- pushed the branch or commit to GitHub
- refreshed the EC2 repo and containers

This guide focuses on the EC2 steps that are easy to miss after a dbt or Airflow code change.

## 1. Refresh EC2 Code

Run from the EC2 repo root:

```bash
git fetch origin
git checkout <your-branch>
git pull --ff-only origin <your-branch>
git rev-parse HEAD
docker compose up -d airflow-postgres dw-postgres mlflow-postgres mlflow
docker compose up -d redis
docker compose up airflow-init
docker compose up -d --build --force-recreate airflow-webserver airflow-scheduler airflow-worker-tier1 airflow-worker-tier2
docker compose ps
docker compose exec airflow-webserver airflow dags list-import-errors
docker compose exec airflow-webserver airflow dags list
```

Expected result:

- `git rev-parse HEAD` matches the commit you validated locally
- Airflow parses without import errors

## 2. Re-apply Runtime Airflow Variables

Run these after a release so Airflow Variables stay aligned with the current dbt DAG contract:

```bash
docker compose exec airflow-webserver airflow variables set DBT_THREADS 2
docker compose exec airflow-webserver airflow variables set DBT_FEATURE_REBUILD_LOOKBACK_MINUTES 120
docker compose exec airflow-webserver airflow variables set DBT_FEATURE_BUILD_TEST_SELECTOR hf_feature_smoke_tests
docker compose exec airflow-webserver airflow variables set DBT_STATION_STATUS_HOTPATH_TEST_SELECTOR hf_station_status_smoke_tests
docker compose exec airflow-webserver airflow variables set DBT_QUALITY_TEST_SELECTOR hourly_quality_gate_tests
docker compose exec airflow-webserver airflow variables set DBT_DEEP_QUALITY_TEST_SELECTOR daily_deep_quality_tests
```

## 3. Mandatory Feature Reconcile After Feature-Label Logic Changes

If the release changes any of the following, you must run a long-window feature reconcile before trusting `dbt_quality_hourly`:

- `dbt/bikeshare_dbt/models/features/feat_station_snapshot_5min.sql`
- feature label maturity logic
- `target_*_t30` logic
- `y_stockout_*_30` logic
- reference tests for `feat_station_snapshot_5min`

Why:

- the EC2 warehouse may still contain previously materialized feature rows built with the old logic
- `dbt_quality_hourly` validates persisted feature rows, not just the SQL text currently on disk

Use a temporary 7-day rebuild window:

```bash
docker compose exec airflow-webserver airflow variables set DBT_FEATURE_REBUILD_LOOKBACK_MINUTES 10080
docker compose exec airflow-webserver airflow dags trigger dbt_feature_build_5min
```

Wait for `dbt_feature_build_5min` to finish successfully, then restore steady-state:

```bash
docker compose exec airflow-webserver airflow variables set DBT_FEATURE_REBUILD_LOOKBACK_MINUTES 120
docker compose exec airflow-webserver airflow dags trigger dbt_quality_hourly
```

## 4. Quick Verification Queries

If the release touched feature labels, verify that recent rows are no longer stuck on old values:

```bash
docker compose exec dw-postgres psql -U velib -d velib_dw -c "
SELECT dt,
       count(*) AS total_rows,
       count(*) FILTER (WHERE target_bikes_t30 IS NOT NULL) AS target_bikes_rows,
       count(*) FILTER (WHERE target_docks_t30 IS NOT NULL) AS target_docks_rows,
       count(*) FILTER (WHERE y_stockout_bikes_30 IS NOT NULL) AS bikes_label_rows,
       count(*) FILTER (WHERE y_stockout_docks_30 IS NOT NULL) AS docks_label_rows
FROM analytics.feat_station_snapshot_5min
WHERE city = 'paris'
  AND dt >= to_char(now() at time zone 'UTC' - interval '3 hours', 'YYYY-MM-DD-HH24-MI')
GROUP BY dt
ORDER BY dt DESC
LIMIT 36;
"
```

If `dbt_quality_hourly` still fails on `feat_station_snapshot_5min_targets_match_t30_reference_source`, compare one failing window directly:

```bash
docker compose exec dw-postgres psql -U velib -d velib_dw -c "
WITH source_rows AS (
    SELECT
        city,
        station_id,
        snapshot_bucket_at_utc,
        num_bikes_available AS bikes,
        num_docks_available AS docks
    FROM analytics.int_station_status_enriched
    WHERE snapshot_bucket_at_utc >= now() at time zone 'UTC' - interval '2 hours'
),
expected_targets AS (
    SELECT
        cur.city,
        cur.station_id,
        cur.snapshot_bucket_at_utc,
        max(fut.snapshot_bucket_at_utc) AS expected_target_snapshot_bucket_at_utc
    FROM source_rows cur
    LEFT JOIN source_rows fut
        ON cur.city = fut.city
       AND cur.station_id = fut.station_id
       AND fut.snapshot_bucket_at_utc > cur.snapshot_bucket_at_utc
       AND fut.snapshot_bucket_at_utc <= cur.snapshot_bucket_at_utc + interval '30 minutes'
    GROUP BY cur.city, cur.station_id, cur.snapshot_bucket_at_utc
)
SELECT *
FROM expected_targets
ORDER BY snapshot_bucket_at_utc DESC
LIMIT 20;
"
```

## 5. Operator Rule

If EC2 quality fails immediately after a release and the failure points to a persisted feature consistency test, first suspect:

1. the EC2 host is not on the expected commit
2. Airflow Variables still hold old selectors or thread settings
3. the feature table has not been reconciled over a long enough lookback window

Do not assume a fresh code deploy automatically rewrites historical incremental feature rows.
