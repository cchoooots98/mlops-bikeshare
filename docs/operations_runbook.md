# Operations Runbook

## Purpose

This runbook covers SLA objectives, daily operations, monitoring signals, alert triage, incident response, and evidence requirements for the dual-target production platform running bikes and docks as independent targets across EC2 data engineering and AWS serving planes.

## 1. Service Objectives (SLA)

This section defines minimum operating thresholds for the dual-target production platform.

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

## 2. Production Admission Gate

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

Required evidence for a green release:
- gate output
- package manifest
- deployment state before and after promotion
- dashboard or CloudWatch confirmation for the promoted target

## 3. Service Inventory

Always-on data plane:
- EC2 host
- Docker Compose
- Airflow
- Postgres
- dbt
- serving DAGs for prediction, quality backfill, metrics publish, and PSI publish
- dashboard

Serving plane:
- S3
- ECR
- SageMaker staging/prod endpoints
- CloudWatch dashboard and alarms
- SNS
- router lambda

## 4. Daily Checks

Every morning:

1. Confirm the EC2 host is reachable.
2. Confirm Docker Compose services are healthy.
3. Confirm Airflow scheduler and webserver are healthy.
4. Confirm dbt freshness and recent DAG success.
5. Check CloudWatch for the impacted environment and target only:
   - `PredictionHeartbeat`
   - `ModelLatency`
   - `Invocation5XXErrors`
   - `PR-AUC-24h`
   - `F1-24h`
   - `PSI`
6. Confirm the dashboard reflects the expected prod versions.

### Monitoring Dimensions

All custom metric checks must filter by:
- `Environment`
- `EndpointName`
- `City`
- `TargetName`

### Monitoring Signals

Custom metrics in `Bikeshare/Model`:
- `PR-AUC-24h`
- `F1-24h`
- `ThresholdHitRate-24h`
- `Samples-24h`
- `PredictionHeartbeat`
- `PSI`

Native metrics in `AWS/SageMaker`:
- `ModelLatency`
- `Invocation5XXErrors`
- `Invocation4XXErrors`
- `Invocations`

## 5. Alert Triage

### Dashboard and Alarm Expectations

The formal AWS stack should expose:
- one CloudWatch dashboard for the environment
- latency and 5xx alarms for each formal endpoint
- PR-AUC, F1, PredictionHeartbeat, and PSI alarms for each formal endpoint
- one SNS topic for monitoring notifications

### First Triage Steps

1. Identify the affected target.
2. Identify the affected environment.
3. Check the dashboard and alarms for that target and environment only.
4. Confirm the deployment state points to the expected endpoint and package.
5. Decide whether to stabilize with rollback.

## 6. Incident Types and Response

### First Response

1. Classify the incident:
   - data freshness
   - training or quality
   - serving or latency
   - deployment or state mismatch
2. Stabilize the service first.
3. Use rollback if prod is degraded.

### Data Freshness Incidents

Symptoms:
- missing predictions
- missing quality shards
- delayed dbt or Airflow runs

Checks:
- Airflow DAG success
- serving DAG success for prediction, quality backfill, metrics publish, and PSI publish
- latest feature table timestamps
- latest `PredictionHeartbeat`
- latest prediction and quality partitions

Immediate action:
- restore the data pipeline first
- do not promote new models while freshness is degraded

### Quality Incidents

Symptoms:
- low `PR-AUC-24h`
- low `F1-24h`
- high `PSI`

Checks:
- verify the dimensions include the correct target
- compare current prod package with previous prod package
- inspect the latest package manifest and training run

Immediate action:
- if prod quality is materially degraded, roll back the affected target

### Serving Incidents

Symptoms:
- `Invocation5XXErrors > 0`
- high latency
- endpoint not `InService`

Checks:
- SageMaker endpoint status
- CloudWatch latency and error metrics
- router lambda behavior
- deployment state endpoint name

Immediate action:
- prefer rollback over hot patching prod

## 7. Retraining Procedure

1. Run offline retraining for one target.
2. Save the package manifest and run ID.
3. Export and deploy the package to staging.
4. Observe staging for 24 hours.
5. Run gate checks with explicit `--environment staging`.
6. Promote only if quality, latency, heartbeat, and drift criteria pass.

## 8. Promotion Rules

- Promote bikes and docks independently.
- Do not promote without saved gate output.
- Production promotion must always use target-specific deployment state.
- Promotion evidence must include:
  - run ID
  - package manifest path
  - model version
  - gate output
  - resulting `production.json`

## 9. Rollback Policy and Procedure

### Policy

- Roll back by target, never globally.
- Roll back to the last approved production deployment state.
- Rollback must not retrain the model.
- Recovery target is minutes, not hours.

### When to Rollback

- 5xx errors appear
- latency breaches the critical threshold
- heartbeat drops below the required cadence
- quality metrics fall below the approved threshold
- PSI crosses the approved warning threshold and the service is degrading
- deployment state points to the wrong package or endpoint

### Rollback Procedure

1. Identify the impacted target.
2. Locate the current `production.json` and the approved `previous_prod.json`.
3. Execute `pipelines.rollback`.
4. Verify the endpoint, dashboard, and restored deployment state.
5. Record the incident and attach evidence.

## 10. Evidence to Keep

From incident response:
- alarm name and time
- target affected
- environment affected
- endpoint affected
- current package and previous package
- CloudWatch dashboard or metric export
- rollback command used
- resulting state file
- follow-up action

From monitoring:
- target name
- environment
- endpoint name
- time window
- relevant metric screenshots or exports
- dashboard name
- deployment state before and after action
- rollback command if used

From promotion:
- gate output
- package manifest
- deployment state before and after promotion
- dashboard or CloudWatch confirmation for the promoted target

## 11. Review Cadence

### Daily

- Operator review of heartbeat, latency, and errors.
- Confirm all daily checks pass (see Section 4).

### Weekly

- Quality and drift review.
- Terraform drift.
- IAM and secret posture.
- CloudWatch dashboard health.
- SNS routing test.
- Staging hygiene.
- Dashboard correctness for bikes and docks.

### Monthly

- Cost review.
- Incident summary.
- Infrastructure drift.
- Dependency drift.
