# Security & Compliance (Step 0)

## Accounts & Roles
- **AWS Account**: 387706002632 (example)
- **Region**: ca-central-1
- **Terraform Backend**
  - S3: `mlops-tfstate-387706002632-ca-central-1` (versioning + SSE-S3)
  - DynamoDB: `mlops-tflock`
- **GitHub OIDC**
  - OIDC Provider: `token.actions.githubusercontent.com`
  - Role (Step 0 minimal): `${project}-${env}-gh-actions`
  - Trust restriction: `repo:${github_owner}/${github_repo}:ref:refs/heads/${github_branch}`
  - Permissions: *no inline permissions at Step 0 (assume-only)*

## Branch Protection
- `main` protected: PR required, status checks required, conversation resolution required.

## Local Dev Environment
- Windows + PowerShell
- VS Code, Git, AWS CLI v2, Terraform ≥ 1.7, Python ≥ 3.10
- Dependency locking via `pip-tools`:
  - `requirements.in` + `requirements-dev.in` → `pip-compile` → `pip-sync`

## Operational Checks
- `aws sts get-caller-identity` passes with `$AWS_PROFILE=Shirley`
- GitHub Actions successfully assumes role and prints caller identity.
- `terraform init -reconfigure` + `validate` succeed under `infra/terraform/`.
