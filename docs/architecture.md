# System Architecture

## Overview
The platform is organized around three explicit boundaries:
- data production: Python ingestion + dbt/Postgres
- model production: offline training + MLflow + local model packages
- model activation: deployment-state files and environment-specific deploy scripts

## Data Flow
1. Python ingestion writes raw payloads and `public.stg_*`.
2. dbt builds curated and feature tables in `analytics.*`.
3. Training reads `analytics.feat_station_snapshot_5min`.
4. Inference reads `analytics.feat_station_snapshot_latest`.
5. Training emits a package directory under `model_dir/packages/...`.
6. Deployment writes an activation record under `model_dir/deployments/*.json`.

## Model Package
Each model package has this fixed shape:

```text
<package_dir>/
  package_manifest.json
  model/
  artifacts/
```

`package_manifest.json` is the static source of truth for:
- target definition
- threshold
- feature contract version
- feature order
- model identity
- registry metadata

## Deployment State
Deployment-state JSON is the dynamic source of truth for:
- active environment
- active package directory
- active registered model/version
- last activation timestamp

Local inference uses `model_dir/deployments/local.json` by default.

## AWS Direction
AWS scripts now work from model packages and deployment-state records:
- export package to tar/S3
- deploy package to SageMaker
- promote deployment state between environments

This keeps local development and future AWS serving on the same package contract.
