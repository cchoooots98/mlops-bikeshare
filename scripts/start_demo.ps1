<#
  Start (create or update) the prod endpoint for demo in a CI-parity way.
  Priority:
    1) If ECR + S3 + ROLE + INSTANCE are available -> call the SAME Python deploy script used by GA.
    2) Else reuse the latest EndpointConfig created previously by GA (fallback).
  Finally publish one heartbeat metric to warm up dashboards.
#>

param(
  [string]$Region        = $env:AWS_REGION,          # e.g., ca-central-1
  [string]$EndpointName  = $(if ($env:SM_ENDPOINT) { $env:SM_ENDPOINT } else { $env:ENDPOINT_NAME }),  # prefer SM_ENDPOINT, else ENDPOINT_NAME
  [string]$City          = $env:CITY                 # e.g., nyc
)

$ErrorActionPreference = "Stop"
if (-not $Region -or $Region -eq "") { $Region = "ca-central-1" }
if (-not $EndpointName -or $EndpointName -eq "") { $EndpointName = "bikeshare-prod" }
if (-not $City -or $City -eq "") { $City = "nyc" }
$env:AWS_PAGER = ""

Write-Host "INFO: Target endpoint '$EndpointName' in region '$Region'"

# --- If endpoint already exists and is InService, just publish heartbeat and exit. ---
try {
  $status = aws sagemaker describe-endpoint --region $Region --endpoint-name $EndpointName --query 'EndpointStatus' --output text 2>$null
} catch { $status = "MISSING" }

if ($status -eq "InService") {
  Write-Host "INFO: Endpoint already InService. Publishing heartbeat..."
  $payload = @"
[
  {
    "MetricName": "PredictionHeartbeat",
    "Value": 1,
    "Unit": "Count",
    "Dimensions": [
      { "Name": "EndpointName", "Value": "$EndpointName" },
      { "Name": "City", "Value": "$City" }
    ]
  }
]
"@
  $payload | Out-File -FilePath .\_cw_boot.json -Encoding ascii
  aws cloudwatch put-metric-data --region $Region --namespace "Bikeshare/Model" --metric-data file://_cw_boot.json
  Remove-Item .\_cw_boot.json -Force
  Write-Host "SUCCESS: Demo endpoint ready: $EndpointName"
  exit 0
}

# --- Decide path: (A) CI-parity deploy via Python SDK, or (B) reuse latest EndpointConfig ---
$ECR  = $env:ECR_IMAGE_URI
$TAR  = $env:S3_MODEL_TAR; if (-not $TAR -or $TAR -eq "") { $TAR = $env:DEFAULT_S3_MODEL_TAR }
$ROLE = $env:SM_EXECUTION_ROLE_Arn
$INST = $env:INSTANCE_TYPE; if (-not $INST -or $INST -eq "") { $INST = "ml.m5.large" }

if ($ECR -and $TAR -and $ROLE -and $INST) {
  Write-Host "INFO: Using CI-parity deploy (Python SDK) with:"
  Write-Host "      Image: $ECR"
  Write-Host "      Model tar: $TAR"
  Write-Host "      Role: $ROLE"
  Write-Host "      Instance: $INST"

  # Comment: Call the same script GA uses (blue/green single-variant).
  python pipelines/deploy_via_sagemaker_sdk.py `
    --endpoint-name "$EndpointName" `
    --role-arn "$ROLE" `
    --image-uri "$ECR" `
    --model-data "$TAR" `
    --instance-type "$INST" `
    --region "$Region"

} else {
  Write-Host "INFO: Missing one of ECR_IMAGE_URI / S3_MODEL_TAR / SM_EXECUTION_ROLE_Arn / INSTANCE_TYPE."
  Write-Host "INFO: Falling back to reuse the latest EndpointConfig created by GA."

  # Comment: Pick the most recent EndpointConfig that contains the endpoint name.
  $cfgJson = aws sagemaker list-endpoint-configs --region $Region --name-contains $EndpointName `
    --query 'sort_by(EndpointConfigs, &CreationTime)[-1].EndpointConfigName' --output json 2>$null
  $cfgName = ""
  if ($LASTEXITCODE -eq 0 -and $cfgJson) {
    $cfgName = (ConvertFrom-Json $cfgJson)
  }

  if (-not $cfgName -or $cfgName -eq "") {
    Write-Host "ERROR: No EndpointConfig found for '$EndpointName'."
    Write-Host "HINT: Run the Promote_Production workflow once to create the initial config,"
    Write-Host "      or set ECR_IMAGE_URI / S3_MODEL_TAR / SM_EXECUTION_ROLE_Arn / INSTANCE_TYPE envs and rerun."
    exit 1
  }

  Write-Host "INFO: Reusing EndpointConfig '$cfgName'"

  # Comment: Create or update endpoint with the chosen config.
  $existingCount = aws sagemaker list-endpoints --region $Region --name-contains $EndpointName `
    --query "length(Endpoints[?EndpointName=='$EndpointName'])" --output text 2>$null

  if ($existingCount -eq "0") {
    aws sagemaker create-endpoint --region $Region --endpoint-name $EndpointName --endpoint-config-name $cfgName | Out-Null
  } else {
    aws sagemaker update-endpoint --region $Region --endpoint-name $EndpointName --endpoint-config-name $cfgName | Out-Null
  }
}

Write-Host "INFO: Waiting for endpoint to be InService..."
aws sagemaker wait endpoint-in-service --region $Region --endpoint-name $EndpointName

# --- Publish one heartbeat metric so dashboards are not empty. ---
$payload2 = @"
[
  {
    "MetricName": "PredictionHeartbeat",
    "Value": 1,
    "Unit": "Count",
    "Dimensions": [
      { "Name": "EndpointName", "Value": "$EndpointName" },
      { "Name": "City", "Value": "$City" }
    ]
  }
]
"@
$payload2 | Out-File -FilePath .\_cw_boot.json -Encoding ascii
aws cloudwatch put-metric-data --region $Region --namespace "Bikeshare/Model" --metric-data file://_cw_boot.json
Remove-Item .\_cw_boot.json -Force

Write-Host "SUCCESS: Demo endpoint ready: $EndpointName"
