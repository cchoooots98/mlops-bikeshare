# Evidence Capture Template

## Purpose
Use this template whenever a phase claims to be complete.

Do not mark a phase green until the evidence exists.

## Recommended Folder
- `docs/evidence/<YYYY-MM-DD>-<phase-name>/`

## Minimum Evidence Pack
| Item | Example |
|---|---|
| command log | terminal output saved to `commands.txt` |
| screenshots | CloudWatch dashboard, endpoint status, Airflow UI |
| state files | `staging.json`, `production.json`, `previous_prod.json` |
| artifact identity | package manifest path, run ID, model version |
| pass/fail note | one short summary stating whether the gate passed |

## Phase-Specific Evidence
### Local baseline
- `pytest -q`
- `pip check`
- `docker compose ps`

### dbt feature chain
- ingest command outputs
- `dbt run`
- `dbt test`
- SQL validation result

### Training
- bikes package manifest
- docks package manifest
- MLflow run IDs

### Local orchestration loop
- predictor output
- quality shard key
- metrics dry-run output

### EC2 always-on
- 72-hour Airflow success window
- dbt freshness checks
- restart recovery evidence

### Terraform
- bootstrap output values for backend bucket and region
- `terraform plan` summary
- `terraform apply` summary
- output values for bucket, SNS topic, dashboard, router lambda

### Staging and production
- endpoint `InService` evidence
- gate output with explicit `--environment staging`
- promotion output
- rollback output and restored state

## Required Metadata
- phase name
- target name if applicable
- environment
- date and UTC time window
- operator name
