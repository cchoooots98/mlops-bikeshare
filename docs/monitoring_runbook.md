# Monitoring Runbook

## Purpose
This runbook explains how to monitor and triage the dual-target platform.

## Formal Dimensions
All custom metric checks must filter by:
- `Environment`
- `EndpointName`
- `City`
- `TargetName`

## Main Signals
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

## Dashboard And Alarm Expectations
The formal AWS stack should expose:
- one CloudWatch dashboard for the environment
- latency and 5xx alarms for each formal endpoint
- PR-AUC, F1, PredictionHeartbeat, and PSI alarms for each formal endpoint
- one SNS topic for monitoring notifications

## First Triage Steps
1. Identify the affected target.
2. Identify the affected environment.
3. Check the dashboard and alarms for that target and environment only.
4. Confirm the deployment state points to the expected endpoint and package.
5. Decide whether to stabilize with rollback.

## Data And Freshness Incidents
Symptoms:
- missing predictions
- missing quality shards
- delayed dbt or Airflow runs

Checks:
- Airflow DAG success
- latest feature table timestamps
- latest `PredictionHeartbeat`
- latest prediction and quality partitions

Immediate action:
- restore the data pipeline first
- do not promote new models while freshness is degraded

## Quality Incidents
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

## Serving Incidents
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

## Evidence To Keep
- target name
- environment
- endpoint name
- time window
- relevant metric screenshots or exports
- dashboard name
- deployment state before and after action
- rollback command if used
