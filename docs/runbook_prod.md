# Production Runbook

## Scope
This runbook covers the formal operating model:
- EC2 data engineering plane
- AWS serving plane
- bikes and docks targets running independently

For command details, use:
- [deployment_guide.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/deployment_guide.md)

## Service Inventory
Always-on data plane:
- EC2 host
- Docker Compose
- Airflow
- Postgres
- dbt
- dashboard

Serving plane:
- S3
- ECR
- SageMaker staging/prod endpoints
- CloudWatch dashboard and alarms
- SNS
- router lambda

## Daily Checks
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

## Retraining Procedure
1. Run offline retraining for one target.
2. Save the package manifest and run ID.
3. Export and deploy the package to staging.
4. Observe staging for 24 hours.
5. Run gate checks with explicit `--environment staging`.
6. Promote only if quality, latency, heartbeat, and drift criteria pass.

## Promotion Rules
- Promote bikes and docks independently.
- Do not promote without saved gate output.
- Production promotion must always use target-specific deployment state.
- Promotion evidence must include:
  - run ID
  - package manifest path
  - model version
  - gate output
  - resulting `production.json`

## Incident Response
### First response
1. Classify the incident:
   - data freshness
   - training or quality
   - serving or latency
   - deployment or state mismatch
2. Stabilize the service first.
3. Use rollback if prod is degraded.

### When to rollback
- 5xx errors appear
- latency breaches the critical threshold
- heartbeat drops below the required cadence
- quality metrics fall below the approved threshold
- PSI crosses the approved warning threshold and the service is degrading
- deployment state points to the wrong package or endpoint

### Rollback procedure
1. Identify the impacted target.
2. Locate the current `production.json` and the approved `previous_prod.json`.
3. Execute `pipelines.rollback`.
4. Verify the endpoint, dashboard, and restored deployment state.
5. Record the incident and attach evidence.

## Evidence To Capture
- alarm name and time
- target affected
- environment affected
- endpoint affected
- current package and previous package
- CloudWatch dashboard or metric export
- rollback command used
- resulting state file
- follow-up action

## Weekly Review
- Terraform drift
- IAM and secret posture
- CloudWatch dashboard health
- SNS routing test
- staging hygiene
- dashboard correctness for bikes and docks
