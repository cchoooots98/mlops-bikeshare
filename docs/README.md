# Docs Index

## Purpose

This file is the document map for the repository.

Use it to answer one question quickly: which document should I open next?

## Authority Map

| Document | Audience | Authority |
|---|---|---|
| [execution_guide.md](execution_guide.md) | reviewer / builder | High-level day-by-day summary of the build sequence. |
| [deployment_guide.md](deployment_guide.md) | operator / deployer | Command authority for local, EC2, Terraform, staging, promote, and rollback. |
| [operations_runbook.md](operations_runbook.md) | day-2 operator | SLA thresholds, monitoring signals, daily checks, incidents, and rollback procedure. |
| [security.md](security.md) | operator / reviewer | Secrets, IAM, backend, and compliance checks. |
| [data_pipeline.md](data_pipeline.md) | builder / analyst | Data contract, warehouse model, and feature store. |
| [ml_lifecycle.md](ml_lifecycle.md) | builder / reviewer | Training, evaluation, and model packaging. |
| [dashboard.md](dashboard.md) | operator / reviewer | Dashboard spec and implementation. |

## Start Here

- Active build or rebuild: [execution_guide.md](execution_guide.md)
- Formal deployment commands: [deployment_guide.md](deployment_guide.md)
- Day-2 operations and incidents: [operations_runbook.md](operations_runbook.md)

## Core Technical Documents

- [architecture.md](architecture.md): system layout, target isolation, and operating split
- [data_pipeline.md](data_pipeline.md): data contract, warehouse model, and feature store
- [ml_lifecycle.md](ml_lifecycle.md): training, evaluation, and model packaging
- [evidence_capture_template.md](evidence_capture_template.md): standard evidence pack template for each phase gate

## Operations and Validation Documents

- [deployment_guide.md](deployment_guide.md): local, EC2, Terraform, staging, production, and rollback commands
- [operations_runbook.md](operations_runbook.md): SLA, monitoring, daily checks, and incident response
- [dashboard.md](dashboard.md): dashboard behavior and target-aware expectations
- [security.md](security.md): secrets, IAM, Terraform backend, and compliance posture
- [cicd.md](cicd.md): CI/CD pipeline and packaging flow

## Internal Notes

Use `docs/plan_detail/` for owner-operator working material only.

Historical plans in that folder are background context, not current execution entrypoints.

## File Structure

```
docs/
├── README.md                     — document index
├── architecture.md               — system architecture and Mermaid diagrams
├── data_pipeline.md              — data contract, warehouse model, feature store
├── ml_lifecycle.md               — training, evaluation, model packaging
├── deployment_guide.md           — command authority for all environments
├── operations_runbook.md         — SLA, monitoring, incident response
├── dashboard.md                  — dashboard spec and implementation
├── execution_guide.md            — 10-day build sequence
├── cicd.md                       — CI/CD and packaging flow
├── security.md                   — security and compliance
├── cheatsheet.md                 — command quick reference
└── evidence_capture_template.md  — phase evidence template
```
