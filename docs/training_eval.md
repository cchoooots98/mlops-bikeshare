# Training And Evaluation

## Contract
- Offline training consumes `analytics.feat_station_snapshot_5min`.
- Feature order is defined by `src/features/schema.py -> FEATURE_COLUMNS`.
- Training outputs a local model package:
  - `package_manifest.json`
  - `model/`
  - `artifacts/`

## Training Responsibilities
`src.training.train` is responsible for:
- loading Postgres feature slices
- validating feature and label contracts
- temporal split
- class-balance checks for train and validation
- model fit and evaluation
- MLflow logging
- local model package creation

## Guardrails
- Training fails if train or validation slices are empty.
- Training fails if either split contains only one class.
- Non-nullable features are checked with:
  - per-column missing-rate threshold
  - overall missing-rate threshold
- Threshold selection is validation-only.

## Retrain Responsibilities
`src.orchestration.retrain` is responsible for:
- refreshing dbt feature tables
- checking feature freshness
- running candidate training
- rejecting low-quality candidates
- registering the candidate in MLflow
- updating the local package manifest with registry metadata
- writing a candidate summary to `model_dir/candidates/retrain_summary.json`

It does not perform deployment.

## Inference Responsibilities
- Inference resolves the active package through deployment state.
- It then loads threshold, target metadata, and feature contract from `package_manifest.json`.
- Retrain summaries are not part of inference configuration anymore.
