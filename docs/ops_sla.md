# Operations SLO And SLA

## Scope
This document defines minimum operating thresholds for the dual-target production platform.

## Service Objectives
| Area | Metric | Target |
|---|---|---|
| Endpoint latency p95 | `AWS/SageMaker.ModelLatency` | warning at 200 ms, critical at 300 ms |
| 5xx errors | `AWS/SageMaker.Invocation5XXErrors` | 0 |
| PR-AUC 24h | `Bikeshare/Model.PR-AUC-24h` | at or above 0.70 |
| F1 24h | `Bikeshare/Model.F1-24h` | at or above 0.55 |
| Prediction cadence | `Bikeshare/Model.PredictionHeartbeat` | at least one heartbeat per 15-minute window |
| Drift | `Bikeshare/Model.PSI` | warning at 0.20, critical at 0.30 |

All custom metric evaluations must filter by:
- `Environment`
- `EndpointName`
- `City`
- `TargetName`

## Production Admission Gate
Before promotion, the target-specific staging endpoint must satisfy:
1. `PR-AUC-24h` meets or exceeds `0.70`.
2. `F1-24h` meets or exceeds `0.55`.
3. `ModelLatency p95` stays below the warning threshold.
4. `Invocation5XXErrors` stays at `0`.
5. `PredictionHeartbeat` is continuous for the observation window.
6. `PSI` stays below the warning threshold or has an approved waiver.

Formal gate commands must pass `--environment staging`.
The 24-hour staging observation window should be fed by the EC2-hosted staging DAG set:
- `staging_prediction_15min`
- `staging_quality_backfill_15min`
- `staging_metrics_publish_hourly`
- `staging_psi_publish_hourly`

## Rollback Policy
- Roll back by target, never globally.
- Roll back to the last approved production deployment state.
- Rollback must not retrain the model.
- Recovery target is minutes, not hours.

## Required Evidence For A Green Release
- gate output
- package manifest
- deployment state before and after promotion
- dashboard or CloudWatch confirmation for the promoted target

## Review Cadence
- Daily: operator review of heartbeat, latency, and errors
- Weekly: quality and drift review
- Monthly: cost, incident summary, infrastructure drift, and dependency drift
