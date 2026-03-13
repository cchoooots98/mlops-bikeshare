# Platform Module

This module contains the shared AWS platform resources for the bikeshare stack:
- S3 data bucket
- Glue catalog database
- ECR repository
- GitHub OIDC role
- SageMaker execution role
- Router Lambda + EventBridge trigger
- SNS-backed CloudWatch alarms for the four formal SageMaker endpoints

Environment-specific entrypoints live under `infra/terraform/envs/dev` and `infra/terraform/envs/prod`.
