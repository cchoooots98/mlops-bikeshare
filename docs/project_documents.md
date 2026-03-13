# Project Documents

## Purpose
This file is the document map for the repository.

Use it to answer one question quickly: which document should I open next?

## Authority Map
| Document | Audience | Authority |
|---|---|---|
| [current_state_to_enterprise_operator_manual.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/plan_detail/current_state_to_enterprise_operator_manual.md) | builder / new operator | The only step-by-step build and validation entrypoint. |
| [execution_guide.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/execution_guide.md) | reviewer / builder | High-level day-by-day summary of the build sequence. |
| [deployment_guide.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/deployment_guide.md) | operator / deployer | Command authority for local, EC2, Terraform, staging, promote, and rollback. |
| [runbook_prod.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/runbook_prod.md) | day-2 operator | Daily checks, incidents, and rollback procedure. |
| [monitoring_runbook.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/monitoring_runbook.md) | operator / reviewer | Monitoring signals, alarm triage, and evidence to keep. |
| [ops_sla.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/ops_sla.md) | operator / reviewer | Formal thresholds, admission criteria, and recovery targets. |
| [security_compliance.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/security_compliance.md) | operator / reviewer | Secrets, IAM, backend, and compliance checks. |
| [evidence_capture_template.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/evidence_capture_template.md) | builder / reviewer | Standard evidence pack template for each phase gate. |

## Start Here
- Active build or rebuild: [current_state_to_enterprise_operator_manual.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/plan_detail/current_state_to_enterprise_operator_manual.md)
- Quick summary of the sequence: [execution_guide.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/execution_guide.md)
- Formal deployment commands: [deployment_guide.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/deployment_guide.md)

## Core Technical Documents
- [architecture.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/architecture.md): system layout, target isolation, and operating split
- [data_contract.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/data_contract.md): feature and data interface expectations
- [data_model.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/data_model.md): warehouse and dbt-oriented model structure
- [feature_store.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/feature_store.md): feature-serving assumptions and storage layout
- [training_eval.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/training_eval.md): offline evaluation approach
- [model_card.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/model_card.md): model behavior and governance summary

## Operations And Validation Documents
- [deployment_guide.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/deployment_guide.md): local, EC2, Terraform, staging, production, and rollback commands
- [runbook_prod.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/runbook_prod.md): day-2 operations and incident response
- [monitoring_runbook.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/monitoring_runbook.md): monitoring metrics, alert handling, and investigation flow
- [dashboard_spec.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/dashboard_spec.md): dashboard behavior and target-aware expectations
- [ops_sla.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/ops_sla.md): operating thresholds and service objectives
- [security_compliance.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/security_compliance.md): secrets, IAM, Terraform backend, and compliance posture
- [evidence_capture_template.md](C:/Career/selfGrowth/projects/mlops-bikeshare-202508/docs/evidence_capture_template.md): evidence pack template for phase completion

## Internal Notes
Use `docs/plan_detail/` for owner-operator working material only.

Historical plans in that folder are background context, not current execution entrypoints.
