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

- Bootstrap the remote backend once, then use those backend values when initializing the live stack.
- Do not assume any repo default `aws_profile` value is correct for your account.
- Keep secrets out of checked-in Terraform files.

## AWS Serving

- Use one long-lived Terraform platform stack for this single-account project.
- Treat `staging` and `production` as model-serving environments, not separate Terraform stacks.
- Use target-specific endpoints, alarms, and deployment state.
- Restrict rollback and promote permissions to approved operators or CI roles.

## Required Periodic Checks

- review IAM policies
- review Terraform drift
- review S3 lifecycle and retention
- review CloudWatch alarm routing
- review SNS topic subscription status or manual publish path
- review local secret handling
