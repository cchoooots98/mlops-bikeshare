# Platform Module

This module contains the shared AWS platform resources for the bikeshare stack:
- S3 data bucket
- Glue catalog database
- ECR repository
- GitHub OIDC role
- SageMaker execution role
- Router Lambda
- SNS-backed CloudWatch alarms for the four formal SageMaker endpoints

The long-lived platform entrypoint lives under `infra/terraform/live`.
Model-serving separation remains `staging` and `production` at the deployment-state and SageMaker endpoint layers.
Production scheduling lives on the EC2/Airflow data plane, not in this Terraform module.
