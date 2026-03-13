# Security And Compliance

## Baseline Requirements
- Python work must run from `.venv`, not a global interpreter.
- Secrets must not be committed to git-tracked files.
- Formal environments must use IAM roles instead of workstation SSO cache mounts.
- Terraform state must remain remote and locked.

## Local Development
- Use `.env.example` as the template.
- Keep real values in `.env`, which is git-ignored.
- Do not rely on global pip installs for formal validation.

## EC2 Always-On Environment
- Attach an instance profile with minimum required permissions.
- Store environment values via secret injection or protected instance configuration.
- Do not mount a developer laptop AWS config into the EC2 stack.

## Terraform Requirements
- `backend.tf` values are environment-specific and must be reviewed before first `terraform init -reconfigure`.
- Do not assume any repo default `aws_profile` value is correct for your account.
- Keep secrets out of checked-in Terraform files.

## AWS Serving
- Separate dev and prod Terraform environments.
- Use target-specific endpoints, alarms, and deployment state.
- Restrict rollback and promote permissions to approved operators or CI roles.

## Required Periodic Checks
- review IAM policies
- review Terraform drift
- review S3 lifecycle and retention
- review CloudWatch alarm routing
- review SNS topic subscription status or manual publish path
- review local secret handling
