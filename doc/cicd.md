# CI/CD — Complete Guide

This document explains how our GitHub Actions workflows are organized for Step 6 and how to operate, troubleshoot, and extend them. All commands below are written for **Windows PowerShell** and are fully commented.

## Overview

We run three workflows under `.github/workflows/`:

- **CI** (`ci.yml`): Lint, test, and security checks on pull requests (and selected pushes).

- **CD • Staging** (`cd_staging.yml`): On push to `main`, deploy the model to the **staging** SageMaker endpoint and smoke test it, with automatic rollback.

- **Promote • Production** (`promote_prod.yml`): Manual approval-gated workflow to promote the model to the **production** SageMaker endpoint, with rollback on failure.


---

## 1) CI (`ci.yml`)

### Triggers

N/A

### Jobs

- **Lint • Test • Security** (id: `python-ci`) — runs-on: `ubuntu-latest`
  - Actions used: actions/checkout@v4, actions/setup-python@v5, actions/cache@v4
- **oidc-smoke** (id: `oidc-smoke`) — runs-on: `ubuntu-latest`
  - Actions used: actions/checkout@v4, aws-actions/configure-aws-credentials@v4
- **terraform-validate** (id: `terraform-validate`) — runs-on: `ubuntu-latest`
  - Actions used: actions/checkout@v4, hashicorp/setup-terraform@v3, aws-actions/configure-aws-credentials@v4

### What CI does

- Checks out the code.
- Sets up Python (3.x).
- Caches pip dependencies.
- Installs development tools: **ruff**, **black**, **pytest**, **bandit**, **pip-audit**.
- Runs lint, format check, unit tests with coverage, and security scans.

### Key tips

- Keep `requirements.txt` and any dev requirements up to date.
- Make `bandit`/`pip-audit` fail the build once initial issues are cleaned up (remove `|| true`).


---

## 2) CD_Staging (`cd_staging.yml`)

### Triggers

N/A

### Jobs

- **Deploy to SageMaker (staging)** (id: `deploy-staging`) — runs-on: `ubuntu-latest`, environment: `staging`
  - Actions used: actions/checkout@v4, aws-actions/configure-aws-credentials@v4, actions/setup-python@v5, aws-actions/configure-aws-credentials@v4

### Environment/Variables/Secrets referenced

- `env` keys: `AWS_REGION, ENDPOINT_NAME, INSTANCE_TYPE, ECR_IMAGE_URI, DEFAULT_S3_MODEL_TAR, SM_EXECUTION_ROLE_Arn`

- `vars` used: `AWS_REGION, ECR_IMAGE_URI, ENDPOINT_STAGING, INSTANCE_STAGING, S3_MODEL_TAR`

- `secrets` used: `AWS_ROLE_TO_ASSUME, SM_EXECUTION_ROLE_Arn`

### What CD (staging) does

- Configures AWS credentials via OIDC (no long-lived keys).

- (Optionally) builds and pushes a BYOC image to ECR.

- Installs Python dependencies required by the deploy script.

- **Deploys/updates** the SageMaker endpoint by calling `pipelines/deploy_via_sagemaker_sdk.py`.

- **Smoke tests** the endpoint using `aws sagemaker-runtime invoke-endpoint`.

- If smoke test fails, **rolls back** to the previous endpoint config.

### Rollback logic

- Before updating, the job captures the current `EndpointConfigName`.
- On failure, it updates the endpoint back to the saved config (blue/green rollback).


---

## 3) Promote_Production (`promote_prod.yml`)

### Triggers

N/A


### Jobs

- **Deploy to SageMaker (prod)** (id: `promote`) — runs-on: `ubuntu-latest`, environment: `production`
  - Actions used: actions/checkout@v4, aws-actions/configure-aws-credentials@v4, actions/setup-python@v5, aws-actions/configure-aws-credentials@v4

### Environment/Variables/Secrets referenced

- `env` keys: `AWS_REGION, ENDPOINT_NAME, INSTANCE_TYPE, ECR_IMAGE_URI, DEFAULT_S3_MODEL_TAR, SM_EXECUTION_ROLE_Arn`

- `vars` used: `AWS_REGION, ECR_IMAGE_URI, ENDPOINT_PROD, INSTANCE_PROD, S3_MODEL_TAR`

- `secrets` used: `AWS_ROLE_TO_ASSUME, SM_EXECUTION_ROLE_Arn`

### What promotion does

- Requires **manual trigger** (`workflow_dispatch`) and can be **approval-gated** via GitHub Environments (`production`).

- Resolves the model artifact (either default S3 URI from repo vars or an override passed as input).

- Deploys a new endpoint config and **updates** the production endpoint.

- Runs a **smoke test**; if it fails, **rolls back** to the previous config.


---

## Operations Runbook

### 1) How to deploy staging manually

```powershell

# Trigger the staging workflow on demand (optional):
# In GitHub → Actions → "CD • Staging" → "Run workflow"

# Or re-run the last successful run from the GitHub UI if you only need a redeploy.

```
### 2) Promote to production

```powershell

# In GitHub → Actions → "Promote • Production" → "Run workflow"
# If the workflow has environment protection on 'production',
# request approval from required reviewers.

```
### 3) Roll back staging/production quickly (CLI)

```powershell

# Get the current endpoint config name (staging example)
aws sagemaker describe-endpoint `
  --endpoint-name bikeshare-staging `
  --query "EndpointConfigName" --output text `
  --region ca-central-1

# Update to a known-good previous config
aws sagemaker update-endpoint `
  --endpoint-name bikeshare-staging `
  --endpoint-config-name bikeshare-staging-previous-config-name `
  --region ca-central-1

```
### 4) View endpoint logs

```powershell

# Tail CloudWatch logs for the endpoint (last 30 minutes)
aws logs tail "/aws/sagemaker/Endpoints/bikeshare-staging" `
  --since 30m --follow --region ca-central-1

```
### 5) Invoke endpoint for a quick smoke test (CLI)

```powershell

# Sample payload for MLflow pyfunc (pandas-split) — adjust columns to your model signature
$payload = @"
{"inputs":{"dataframe_split":{"columns":["station_id","hour","temp_c","dow"],"data":[["72",8,20.5,2]]}}}
"@

aws sagemaker-runtime invoke-endpoint `
  --endpoint-name bikeshare-staging `
  --content-type "application/json" `
  --body $payload `
  --cli-binary-format raw-in-base64-out `
  --region ca-central-1 `
  >(cat)

```
### 6) Local Python smoke script (optional)

```python

# scripts/smoke_invoke.py
# Purpose: Invoke a SageMaker endpoint and print the response.
import argparse, json, boto3

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint-name", required=True, help="Endpoint to call")
    p.add_argument("--region", required=True, help="AWS region, e.g. ca-central-1")
    args = p.parse_args()
    smrt = boto3.client("sagemaker-runtime", region_name=args.region)
    payload = {"inputs":{"dataframe_split":{"columns":["station_id","hour","temp_c","dow"],"data":[["72",8,20.5,2]]}}}
    resp = smrt.invoke_endpoint(EndpointName=args.endpoint_name, ContentType="application/json", Body=json.dumps(payload))
    print(resp["Body"].read().decode("utf-8"))

if __name__ == "__main__":
    main()

```

---

## Configuration

### 1) Required GitHub **Variables**

```text
- `AWS_REGION`
- `AWS_ACCOUNT_ID`
- `ECR_IMAGE_URI`
- `S3_MODEL_TAR`
- `ENDPOINT_STAGING`
- `ENDPOINT_PROD`
- `INSTANCE_STAGING`
- `INSTANCE_PROD`
```
### 2) Required GitHub **Secrets**

```text
- `AWS_ROLE_TO_ASSUME`  # OIDC-assumable role for the GitHub runner
- `SM_EXECUTION_ROLE_Arn`  # SageMaker execution role used by the endpoint
```
### 3) Environments

- **staging**: binds the staging deployment job.
- **production**: add **required reviewers** for approval-gated promotion.


---

## Best Practices

- Use **immutable image tags** (e.g., commit SHA or semver) for reproducible deployments.

- Keep **tests** meaningful and fast; add model-level tests where possible.

- Consider adding **Terraform** steps in CD to continuously enforce infra state.

- Set **CloudWatch Alarms** for endpoint errors/latency and link them to notifications.


---

## Troubleshooting

### `ValidationException` on `--image-uri` (trailing whitespace)

- In PowerShell, ensure no **trailing spaces** after a line continuation backtick.
- Quote the entire URI: `--image-uri "ACCOUNT.dkr.ecr.REGION.amazonaws.com/repo:tag"`.

### `CannotStartContainerError: docker run <image> serve`

- Ensure your image **exists** in ECR and can run `serve` as entrypoint/cmd.
- Verify platform: build the image for `linux/amd64` for `ml.m5.*` instances.
- Confirm that your container **opens port 8080** and responds to `/invocations`.

### OIDC AssumeRole failures

- Check AWS IAM role **trust policy** for `token.actions.githubusercontent.com` with your repo conditions.
- Ensure GitHub workflow uses `aws-actions/configure-aws-credentials@v4` with `id-token: write` permission.

### `MalformedPolicyDocument` or non-ASCII policy files

- Save JSON policy files as **ASCII/UTF-8 without BOM**.
- Avoid smart quotes or hidden characters; use a code editor with whitespace rendering.

### Model signature or payload mismatch

- Align the smoke test payload with your model’s MLflow signature.
- Log a sample input/output during training and reuse that for smoke tests.


---

## Appendices — Annotated workflow snippets

### CI (ci.yml)

_Your current version (for reference):_

```yaml
name: ci

on:
  push:
    branches: ["main"]
  pull_request:
    branches: ["main"]

permissions:
  id-token: write    # needed for OIDC (only used on push to main)
  contents: read

env:
  AWS_REGION: ca-central-1
  AWS_ROLE_ARN: arn:aws:iam::387706002632:role/gh-oidc-deployer

jobs:
# --- Python quality gates run on both PR and push ---
  python-ci:
    name: Lint • Test • Security
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4   # Get repository contents

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'    # Use your project Python version

      - name: Cache pip
        uses: actions/cache@v4      # Speed up dependency installs
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements*.txt') }}
          restore-keys: ${{ runner.os }}-pip-

      - name: Install dependencies (prod + dev)
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install ruff black pytest pytest-cov bandit pip-audit
      
      - name: Install dev tools
        run: |
          python -m pip install --upgrade pip
          pip install ruff==0.5.7 black pytest

      - name: Ruff (auto-fix)
        run: |
          ruff check src pipelines test --fix
      - name: Ruff (enforce)
        run: |
          ruff check src pipelines test --output-format=github

      - name: Black (format check)
        run: |
          black --check src pipelines test

      - name: PyTest 
        env:
          # Ensure Python can import 'src' at repo root
          PYTHONPATH: ${{ github.workspace }}
        run: |
          pytest -q

      - name: Bandit (security SAST)
        run: bandit -q -r src || true   # Set to fail the build later if you prefer

      - name: pip-audit (vuln check)
        run: pip-audit || true          # Start in report-only mode

  # --- Only on push to main: prove OIDC can assume the AWS role ---
  oidc-smoke:
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ env.AWS_ROLE_ARN }}
          aws-region: ${{ env.AWS_REGION }}
          role-session-name: gha-${{ github.run_id }}
          role-duration-seconds: 3600
      
      - name: Verify AWS identity
        run: aws sts get-caller-identity

      - name: Prove we assumed role
        run: aws sts get-caller-identity

  # --- Terraform validate on PR (local backend) and on main (remote backend) ---
  terraform-validate:
    runs-on: ubuntu-latest
    needs: []
    steps:
      - uses: actions/checkout@v4

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: 1.7.5

      # PR: no AWS, no remote backend. Just format + validate
      - name: Terraform fmt & validate (PR)
        if: github.event_name == 'pull_request'
        working-directory: infra/terraform
        run: |
          terraform init -backend=false
          terraform fmt -check -recursive
          terraform validate

      # main: assume role and validate with the real backend.
      - name: Configure AWS credentials (OIDC)
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ env.AWS_ROLE_ARN }}
          aws-region: ${{ env.AWS_REGION }}
          role-session-name: tf-validate-${{ github.run_id }}
          role-duration-seconds: 3600

      - name: Terraform fmt & validate (main)
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        working-directory: infra/terraform
        run: |
          terraform init -reconfigure
          terraform fmt -check -recursive
          terraform validate

```
### CD • Staging (cd_staging.yml)

_Your current version (for reference):_

```yaml
name: CD_Staging

on:
  push:
    branches: [main]
  workflow_dispatch:

env:
  AWS_REGION: ${{ vars.AWS_REGION }}                  # e.g., ca-central-1
  ENDPOINT_NAME: ${{ vars.ENDPOINT_STAGING }}         # e.g., bikeshare-staging
  INSTANCE_TYPE: ${{ vars.INSTANCE_STAGING }}         # e.g., ml.m5.large
  ECR_IMAGE_URI: ${{ vars.ECR_IMAGE_URI }}            # e.g., .../mlflow-pyfunc:3.3.2-v5
  DEFAULT_S3_MODEL_TAR: ${{ vars.S3_MODEL_TAR }}      # default model tar to deploy
  # ---- Secrets ----
  SM_EXECUTION_ROLE_Arn: ${{ secrets.SM_EXECUTION_ROLE_Arn }}

permissions:
  id-token: write
  contents: read

concurrency:
  group: staging-deploy
  cancel-in-progress: false  # Avoid interrupting a live rollout

jobs:
  deploy-staging:
    name: Deploy to SageMaker (staging)
    runs-on: ubuntu-latest
    environment: staging

    steps:
      - name: Checkout
        uses: actions/checkout@v4
      
      - name: Configure AWS (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-region: ${{ env.AWS_REGION }}
          role-to-assume: ${{ secrets.AWS_ROLE_TO_ASSUME }}
          role-session-name: gha-${{ github.run_id }}
      
      - name: Verify AWS identity
        run: aws sts get-caller-identity

      - name: WhoAmI (debug)
        # English: Quick sanity check that credentials work
        run: aws sts get-caller-identity

      # OPTIONAL: If you routinely rebuild/push the BYOC image, add ECR login + docker build/push here.

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install deploy deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt || true
          python - <<'PY'
          import importlib.util, subprocess, sys
          for pkg in ("boto3","botocore"):
              if importlib.util.find_spec(pkg) is None:
                  subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
          PY

      - name: Capture previous endpoint config (for rollback)
        id: prev
        shell: bash
        run: |
          set -euo pipefail
          PREV_CFG=$(aws sagemaker describe-endpoint \
            --endpoint-name "${ENDPOINT_NAME}" \
            --query 'EndpointConfigName' --output text 2>/dev/null || echo "NONE")
          echo "prev_config=$PREV_CFG" >> "$GITHUB_OUTPUT"
          echo "PreviousConfig=$PREV_CFG"

      - name: Deploy (blue/green create-or-update)
        # English: Same parameters as your local PowerShell command
        run: |
          set -euo pipefail
          python pipelines/deploy_via_sagemaker_sdk.py \
            --endpoint-name "${ENDPOINT_NAME}" \
            --role-arn "${SM_EXECUTION_ROLE_Arn}" \
            --image-uri "${ECR_IMAGE_URI}" \
            --model-data "${DEFAULT_S3_MODEL_TAR}" \
            --instance-type "${INSTANCE_TYPE}" \
            --region "${AWS_REGION}"

      - name: Smoke test (use repo script to match schema exactly)
        # English: Reuse your test/smoke_invoke.py to avoid column/order mismatch
        run: |
          set -euo pipefail
          python test/smoke_invoke.py --endpoint-name "${ENDPOINT_NAME}" --region "${AWS_REGION}"
      
      - name: Configure AWS (OIDC) for rollback
        if: failure()
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-region: ${{ env.AWS_REGION }}
          role-to-assume: ${{ secrets.AWS_ROLE_TO_ASSUME }}
          role-session-name: gh-actions-staging-rollback



      - name: Rollback on failure
        if: failure() && steps.prev.outputs.prev_config != 'NONE'
        run: |
          echo "Smoke test failed — rolling back."
          aws sagemaker update-endpoint \
            --endpoint-name "${ENDPOINT_NAME}" \
            --endpoint-config-name "${{ steps.prev.outputs.prev_config }}" \
            --region "${AWS_REGION}"

```
### Promote • Production (promote_prod.yml)

_Your current version (for reference):_

```yaml
name: Promote_Production

on:
  workflow_dispatch:
    inputs:
      override_model_tar:
        description: "Optional S3 model.tar.gz to promote (leave blank to reuse staging model)"
        required: false
        default: ""

permissions:
  id-token: write                 # REQUIRED for GitHub OIDC → AWS
  contents: read

env:
  # ---- Non-secret configuration pulled from repo Variables ----
  AWS_REGION: ${{ vars.AWS_REGION }}                 # e.g., ca-central-1
  ENDPOINT_NAME: ${{ vars.ENDPOINT_PROD }}           # e.g., bikeshare-prod
  INSTANCE_TYPE: ${{ vars.INSTANCE_PROD }}           # e.g., ml.m5.large (or larger)
  ECR_IMAGE_URI: ${{ vars.ECR_IMAGE_URI }}           # e.g., .../mlflow-pyfunc:3.3.2-v5
  DEFAULT_S3_MODEL_TAR: ${{ vars.S3_MODEL_TAR }}     # default model tar if no override
  # ---- Secrets ----
  SM_EXECUTION_ROLE_Arn: ${{ secrets.SM_EXECUTION_ROLE_Arn }}
  


concurrency:
  group: prod-deploy
  cancel-in-progress: false

jobs:
  promote:
    name: Deploy to SageMaker (prod)
    runs-on: ubuntu-latest
    environment: production   # Protect this environment in GitHub for approvals

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Configure AWS (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-region: ${{ env.AWS_REGION }}
          role-to-assume: ${{ secrets.AWS_ROLE_TO_ASSUME }}
          role-session-name: gha-${{ github.run_id }}

      - name: Verify AWS identity
        run: aws sts get-caller-identity
      
      - name: WhoAmI (debug)
        run: aws sts get-caller-identity

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install deploy deps
        # English: Ensure deploy and smoke scripts can import boto3 etc.
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt || true
          python - <<'PY'
          import importlib.util, subprocess, sys
          for pkg in ("boto3","botocore"):
              if importlib.util.find_spec(pkg) is None:
                  subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
          PY

      - name: Resolve model artifact
        id: resolve
        shell: bash
        run: |
          if [ -n "${{ github.event.inputs.override_model_tar }}" ]; then
            echo "model_tar=${{ github.event.inputs.override_model_tar }}" >> "$GITHUB_OUTPUT"
            echo "Using override model tar"
          else
            echo "model_tar=${DEFAULT_S3_MODEL_TAR}" >> "$GITHUB_OUTPUT"
            echo "Using default model tar from repo variables"
          fi

      - name: Save previous endpoint config (for rollback)
        id: prev
        run: |
          PREV_CFG=$(aws sagemaker describe-endpoint \
            --endpoint-name "${ENDPOINT_NAME}" \
            --query 'EndpointConfigName' --output text 2>/dev/null || echo "NONE")
          echo "prev_config=$PREV_CFG" >> "$GITHUB_OUTPUT"
          echo "PreviousConfig=$PREV_CFG"

      - name: Deploy prod (blue/green)
        run: |
          python pipelines/deploy_via_sagemaker_sdk.py \
            --endpoint-name "${ENDPOINT_NAME}" \
            --role-arn  "${SM_EXECUTION_ROLE_Arn}"\
            --image-uri "${ECR_IMAGE_URI}" \
            --model-data "${{ steps.resolve.outputs.model_tar }}" \
            --instance-type "${INSTANCE_TYPE}" \
            --region "${AWS_REGION}"

      - name: Smoke test prod (use repo script to match schema exactly)
        run: |
          set -euo pipefail
          python test/smoke_invoke.py --endpoint-name "${ENDPOINT_NAME}" --region "${AWS_REGION}"

      - name: Configure AWS (OIDC) for rollback
        if: failure()
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-region: ${{ env.AWS_REGION }}
          role-to-assume: ${{ secrets.AWS_ROLE_TO_ASSUME }}
          role-session-name: gh-actions-prod-rollback


      - name: Rollback on failure
        if: failure() && steps.prev.outputs.prev_config != 'NONE'
        run: |
          echo "Smoke test failed — rolling back prod."
          aws sagemaker update-endpoint \
            --endpoint-name "${ENDPOINT_NAME}" \
            --endpoint-config-name "${{ steps.prev.outputs.prev_config }}" \
            --region "${AWS_REGION}"

```