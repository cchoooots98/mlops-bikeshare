# Enterprise Execution Guide

## Purpose
This is the high-level build sequence for the repository.

If you are actively executing commands, use the operator manual as the detailed entrypoint:
- [current_state_to_enterprise_operator_manual.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/plan_detail/current_state_to_enterprise_operator_manual.md)

Use this guide when you want the order and intent without all troubleshooting detail.

## Global Rules
- Use Python `3.11`.
- Use `.venv` for every formal Python command.
- Use `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD` as the formal DB contract.
- Use `MLFLOW_TRACKING_URI=sqlite:///model_dir/mlflow.db` locally.
- Keep bikes and docks isolated in package roots, deployment state, S3 partitions, metrics, and endpoints.
- Do not skip gates.
- Do not claim a phase is complete without an evidence pack.

## Day 1: Local Baseline
Goal:
- Rebuild the local environment from zero and prove it is repeatable.

Main checks:
- `.venv`
- `pip check`
- `pytest -q`
- `docker compose ps`

Pass gate:
- local validation is green
- Airflow and Postgres services are healthy

## Day 2: Runtime Contract
Goal:
- Make `.env`, code, and docs agree on one runtime contract.

Main checks:
- `PG*` variables exist
- local MLflow points to SQLite
- no formal docs depend on `DW_*` or `localhost:5000`

Pass gate:
- runtime settings load without missing-key errors

## Day 3: Dual-Target Structure
Goal:
- Confirm the repository is safe for both bikes and docks.

Main checks:
- package roots are target-aware
- deployment states are target-aware
- metrics and dashboard are target-aware

Pass gate:
- structure tests pass
- no formal path depends on single-file deployment state

## Day 4: dbt Feature Chain
Goal:
- Prove training and online inference both depend on dbt-owned feature tables.

Main checks:
- ingest runs
- `dbt run`
- `dbt test`
- feature tables and labels are non-empty

Pass gate:
- `analytics.feat_station_snapshot_5min`
- `analytics.feat_station_snapshot_latest`
- bikes and docks labels exist

## Day 5: Offline Training
Goal:
- Produce target-specific packages for bikes and docks.

Main checks:
- bikes training
- docks training
- package manifests
- MLflow tags

Pass gate:
- both targets produce valid package manifests

## Day 6: Local Orchestration Loop
Goal:
- Prove the package can drive prediction, quality backfill, and metrics publication.

Main checks:
- activate a local deployment state
- run predictor against a real endpoint
- run quality backfill
- run metrics dry-run

Pass gate:
- prediction and quality partitions include `target=`
- metrics dimensions include `Environment`, `EndpointName`, `City`, `TargetName`

Stop condition:
- if you do not yet have a reachable staging or prod endpoint, stop here and finish Day 8 first

## Day 7: Always-On EC2 Data Plane
Goal:
- Move scheduled data engineering off the laptop.

Main checks:
- EC2 instance role
- Docker Compose
- Airflow
- Postgres
- dbt freshness
- 72-hour run window

Pass gate:
- no workstation dependency remains

## Day 8: Terraform Infrastructure
Goal:
- Create the minimum AWS serving and monitoring infrastructure.

Main checks:
- remote backend is replaced with your real backend before `terraform init`
- provider profile is intentionally set or omitted
- CloudWatch dashboard exists
- quality alarms exist for the four formal endpoints
- SNS topic exists and can receive a test publish
- repo defaults for backend/profile are treated as placeholders to review, not as formal defaults

Pass gate:
- dev and prod `plan`
- dev and prod `apply`
- outputs are recorded in the evidence pack

## Day 9: Staging, Gate, Promote, Rollback
Goal:
- Establish a real release process.

Main checks:
- build inference image
- export and upload bikes and docks packages
- deploy both staging endpoints
- run gate with explicit `--environment staging`
- promote only after the staging gate passes
- run rollback drill with previous production state

Pass gate:
- both staging endpoints are `InService`
- gate output is saved
- rollback completes for one target without affecting the other

## Day 10: Operator Readiness
Goal:
- Make the project executable by someone who did not build it.

Main checks:
- docs are consistent
- commands match the real scripts
- expected outputs are documented
- a new operator can complete one cold-start drill

Pass gate:
- cold-start walkthrough is successful and its feedback is captured

## Weekly Review
- review Terraform drift
- review IAM and secret handling
- review S3 lifecycle and retention
- review CloudWatch alarms and SNS routing
- review staging hygiene
