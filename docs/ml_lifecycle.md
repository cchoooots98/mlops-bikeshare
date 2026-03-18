# ML Lifecycle

## Overview

The ML lifecycle covers offline training on historical feature snapshots, MLflow logging of parameters and artifacts, local model packaging into a versioned directory, deployment state tracking as the dynamic source of truth for active environments, inference that resolves configuration entirely from the deployment state and package manifest, quality monitoring through CloudWatch custom metrics and alarms, and retraining that refreshes features, fits a new candidate, and registers it without performing deployment.

## 1. Training

### Data Source

- Offline training reads from `analytics.feat_station_snapshot_5min`
- Feature order is defined by `src/features/schema.py: FEATURE_COLUMNS`
- Training script: `src/training/train.py`

### Method

- Temporal split: 80% train / 20% validation, with a 60-minute gap to prevent leakage
- Algorithms: XGBoost or LightGBM (configurable via `--model-type`)
- Threshold selection: F-Beta optimization on validation set only

`src.training.train` is responsible for:
- loading Postgres feature slices
- validating feature and label contracts
- temporal split
- class-balance checks for train and validation
- model fit and evaluation
- MLflow logging
- local model package creation

### Guardrails (Training Fails If)

- Training fails if train or validation slices are empty.
- Training fails if either split contains only one class.
- Non-nullable features are checked with:
  - per-column missing-rate threshold
  - overall missing-rate threshold
- Threshold selection is validation-only.

### MLflow Logging

- Tracking URI: `sqlite:///model_dir/mlflow.db` (local default)
- Logs: parameters, metrics, feature importance, threshold, artifact path

### Output Package Structure

```text
model_dir/packages/<target>/<model-name>/<run-id>/
  package_manifest.json   ← static configuration source of truth
  model/                  ← MLflow pyfunc format
  artifacts/
    model_card.md         ← generated from training summary
    eval_summary.json     ← validation metrics
```

## 2. Package Manifest

The `package_manifest.json` is the static source of truth for:
- target definition (bikes or docks)
- decision threshold
- feature contract version
- feature column order
- model identity (run ID, model name)
- MLflow registry metadata

## 3. Retraining

`src.orchestration.retrain` is responsible for:
- refreshing dbt feature tables
- checking feature freshness
- running candidate training
- rejecting low-quality candidates
- registering the candidate in MLflow
- updating the local package manifest with registry metadata
- writing a candidate summary to `model_dir/candidates/retrain_summary.json`

It does not perform deployment.

## 4. Deployment State

The deployment state JSON is the dynamic source of truth for:
- active environment (local/staging/production)
- active package directory
- active registered model version
- last activation timestamp
- active endpoint name

Formal layout:

```text
model_dir/deployments/bikes/local.json
model_dir/deployments/bikes/staging.json
model_dir/deployments/bikes/production.json
model_dir/deployments/docks/local.json
model_dir/deployments/docks/staging.json
model_dir/deployments/docks/production.json
```

## 5. Inference

- Inference resolves the active package through deployment state.
- It then loads threshold, target metadata, and feature contract from `package_manifest.json`.
- Retrain summaries are not part of inference configuration anymore.

## 6. Model Card

Generated model cards belong inside each local model package under:

```text
artifacts/model_card.md
```

The generated card is derived from the training summary and kept next to:
- `package_manifest.json`
- `artifacts/eval_summary.json`
- the packaged MLflow model in `model/`

This keeps offline evaluation artifacts and deployable metadata in one package boundary.

## 7. Target Isolation

- bikes and docks packages must use separate roots
- deployment states must be target-specific
- no shared package or state is valid in dual-target workflows
