# Model Card - paris_y_stockout_bikes_30_xgboost

## Overview
- Use case: Predict short-term (30 min) stockout risk for bikes at bikeshare stations.
- Business objective: Support proactive rebalancing before a station hits low inventory or low dock capacity.
- Owner: MLOps Team
- Date (UTC): 2026-03-24T18:38:56Z

## Data
- City: paris
- Time window (UTC): 2026-03-17 15:10 -> 2026-03-24 16:10
- Sample sizes: train=2036657, valid=492260
- Feature source: `analytics.feat_station_snapshot_5min`
- Feature contract: `v1_dim_weather_aligned`
- Feature list (first 10): minutes_since_prev_snapshot, util_bikes, util_docks, delta_bikes_5m, delta_docks_5m, roll15_net_bikes, roll30_net_bikes, roll60_net_bikes, roll15_bikes_mean, roll30_bikes_mean (total 30)
- Labels: y_stockout_bikes_30

## Modeling
- Model name: paris_y_stockout_bikes_30_xgboost
- Algorithm: xgboost
- Primary metric: PR-AUC (validation) = 0.961
- Train PR-AUC: 0.953; Overfit gap: 0.008 (target < 0.10)
- Threshold (F-beta, beta=2.0): 0.20
  - Precision=0.780, Recall=0.953, F-beta=0.913

## Assumptions And Limitations
- Offline training consumes dbt-owned Postgres feature tables and the Python schema contract in `src/features/schema.py`.
- Neighbor features come from the warehouse radius-based neighbor graph, not a runtime BallTree rebuild.
- Labels are considered mature only when the full future horizon is present in the offline feature table.
- Temporal validation is strictly later than training and separated by an anti-leakage gap.

## Monitoring Plan
- Track PR-AUC, F1, PSI, input freshness, and serving latency after candidate deployment.
- Retraining should create a new candidate and pass staging admission checks before production promotion.

## Reproducibility
- MLflow experiment: bikeshare-local-tunnel
- Run ID: fa5bb7ab2f53452d94768f8a0bd5edf5
- Serving artifact path: `model`
- Training code: `src/training/train.py`
- Evaluation code: `src/training/eval.py`
