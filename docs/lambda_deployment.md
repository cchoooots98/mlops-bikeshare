# AWS Lambda Deployment Guide

This guide describes how to deploy the bikeshare predictor as an AWS Lambda function using a Docker container image.

## Prerequisites

- AWS CLI configured with appropriate credentials
- Docker with buildx support
- An ECR repository created in your AWS account
- Lambda function configured with appropriate IAM role

## Build and Deploy

### 1. Set Environment Variables

```powershell
# PowerShell
$REGION = "ca-central-1"
$ACCOUNT_ID = (aws sts get-caller-identity | ConvertFrom-Json).Account
$REPO = "mlops-bikeshare-predictor"
$IMAGE = "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${REPO}:lambda-amd64"
```

```bash
# Bash
REGION="ca-central-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPO="mlops-bikeshare-predictor"
IMAGE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:lambda-amd64"
```

### 2. Login to ECR

```powershell
# PowerShell
aws ecr get-login-password --region $REGION | docker login `
  --username AWS `
  --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
```

```bash
# Bash
aws ecr get-login-password --region $REGION | docker login \
  --username AWS \
  --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
```

### 3. Create ECR Repository (if not exists)

```powershell
# PowerShell
aws ecr describe-repositories --repository-names $REPO --region $REGION 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
  aws ecr create-repository --repository-name $REPO --region $REGION | Out-Null
}
```

```bash
# Bash
aws ecr describe-repositories --repository-names $REPO --region $REGION 2>/dev/null || \
  aws ecr create-repository --repository-name $REPO --region $REGION
```

### 4. Build and Push Docker Image

```powershell
# PowerShell
docker buildx build `
  --platform linux/amd64 `
  -f Dockerfile.lambda `
  -t $IMAGE `
  --provenance=false `
  --sbom=false `
  --push `
  .
```

```bash
# Bash
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile.lambda \
  -t $IMAGE \
  --provenance=false \
  --sbom=false \
  --push \
  .
```

### 5. Update Lambda Function

```powershell
# PowerShell
$FUNC = "bikeshare-predictor"
aws lambda update-function-code `
  --function-name $FUNC `
  --image-uri $IMAGE `
  --region $REGION
```

```bash
# Bash
FUNC="bikeshare-predictor"
aws lambda update-function-code \
  --function-name $FUNC \
  --image-uri $IMAGE \
  --region $REGION
```

### 6. Invoke the Lambda Function

```powershell
# PowerShell
aws lambda invoke `
  --function-name $FUNC `
  --payload "{}" `
  out.json `
  --region $REGION
```

```bash
# Bash
aws lambda invoke \
  --function-name $FUNC \
  --payload '{}' \
  out.json \
  --region $REGION
```

## Environment Variables

The Lambda function requires the following environment variables (set via Lambda configuration):

- `CITY`: The city to process (e.g., "NYC", "nyc")
- `SM_ENDPOINT`: SageMaker endpoint name (e.g., "bikeshare-staging")
- `BUCKET`: S3 bucket for data storage
- `ATHENA_DATABASE`: Athena database name (e.g., "mlops-bikeshare")
- `ATHENA_OUTPUT`: S3 location for Athena query results
- `ATHENA_WORKGROUP`: Athena workgroup (default: "primary")

These can be set using the AWS Console or CLI:

```bash
aws lambda update-function-configuration \
  --function-name bikeshare-predictor \
  --environment "Variables={
    CITY=NYC,
    SM_ENDPOINT=bikeshare-staging,
    BUCKET=mlops-bikeshare-ACCOUNT-REGION,
    ATHENA_DATABASE=mlops-bikeshare,
    ATHENA_OUTPUT=s3://mlops-bikeshare-ACCOUNT-REGION/athena-results/,
    ATHENA_WORKGROUP=primary
  }" \
  --region $REGION
```

## IAM Permissions

The Lambda execution role needs permissions for:

- S3: Read/Write to the data bucket
- Athena: Execute queries
- SageMaker: InvokeEndpoint
- CloudWatch Logs: Create log groups and streams

## Troubleshooting

### Rate Limiting

If you encounter `TooManyRequestsException`, wait a few moments before invoking again. The Lambda function may still be updating after a code deployment.

### Function Timeout

The default Lambda timeout is 3 seconds. For the predictor, configure at least 900 seconds (15 minutes):

```bash
aws lambda update-function-configuration \
  --function-name bikeshare-predictor \
  --timeout 900 \
  --region $REGION
```

### Memory Issues

If the function runs out of memory, increase it:

```bash
aws lambda update-function-configuration \
  --function-name bikeshare-predictor \
  --memory-size 3072 \
  --region $REGION
```

## Files

- `Dockerfile.lambda` - Lambda container image definition
- `requirements-lambda.txt` - Python dependencies for Lambda
- `src/inference/handler.py` - Lambda entry point
- `.dockerignore` - Files to exclude from Docker build context
