# Model Card — xgboost-y_stockout_bikes_30

## Overview
- **Use case**: Predict short-term (30 min) stockout risk for bikes (stockout) at bikeshare stations.
- **Business objective**: Enable proactive rebalancing to reduce customer dissatisfaction and lost trips.
- **Owner**: Yuanyuan Feng
- **Date**: 2025-09-06 15:10:08Z

## Data
- **City**: nyc
- **Time window (UTC)**: 2025-08-28 00:00 → 2025-09-04 00:00
- **Sample sizes**: train=3592644, valid=904332
- **Feature source**: `features_offline` (Athena external table; partitioned by `city`, `dt`)
- **Feature list (first 10)**: util_bikes, util_docks, delta_bikes_5m, delta_docks_5m, roll15_net_bikes, roll30_net_bikes, roll60_net_bikes, roll15_bikes_mean, roll30_bikes_mean, roll60_bikes_mean (total 25)
- **Labels**: y_stockout_bikes_30

## Modeling
- **Algorithm**: xgboost
- **Primary metric**: PR-AUC (validation) = **0.934**
- **Train PR-AUC**: 0.953; **Overfit gap**: 0.019 (target < 0.10)
- **Threshold (Fβ, β=2.0)**: 0.15
  - Precision=0.723, Recall=0.959, Fβ=0.900

## Assumptions & Limitations
- Assumes 5-minute gridded station status and hourly weather aligned to UTC.
- Neighbor features based on spatial BallTree and inverse-distance weighting.
- Labels defined as stockout if `target_*_t30 ≤ threshold` within 30-minute horizon.
- Temporal split ensures validation comes strictly after training period with a gap to reduce leakage.

## Risks
- **Concept drift** (demand shifts, special events, policy changes).
- **Data delays or outages** (GBFS feed, weather source).
- **Imbalanced labels** — monitor precision/recall and threshold regularly.

## Fairness & Ethics
- Model focuses on station-level operational KPIs; does not infer protected attributes about individuals.
- Ensure equitable service levels across neighborhoods; monitor outcomes for systematic bias.

## Monitoring Plan
- Track PR-AUC (rolling window), precision, recall at deployed threshold.
- Alert on metric degradation or data freshness issues.
- Periodic re-training schedule (e.g., weekly) with backtests.

## Reproducibility
- **MLflow experiment**: bikeshare-step4
- **Run ID**: 101d0d6b948e41c6875e3cab755ba7e6
- **Code**: `training/train.py`, `training/eval.py`
- **Data schema**: `schema.py`

