<#
  Stop (delete) the SageMaker endpoint safely.
  - Idempotent: if the endpoint does not exist, exit 0.
  - Windows-friendly: avoids exposing AWS CLI stderr as a terminating error.
  - All comments are in English.
#>

param(
  # Comment: Region and endpoint can come from env or be passed explicitly.
  [string]$Region = $env:AWS_REGION,
  [string]$Endpoint = $env:SM_ENDPOINT
)

# Comment: Fail on our own PowerShell errors; we handle CLI non-zero codes ourselves.
$ErrorActionPreference = "Stop"

# Comment: Sensible defaults.
if (-not $Region -or $Region -eq "") { $Region = "ca-central-1" }
if (-not $Endpoint -or $Endpoint -eq "") {
  Write-Host "ERROR: Set SM_ENDPOINT env or pass -Endpoint bikeshare-prod"
  exit 1
}

# Comment: Ensure AWS CLI does not use a pager.
$env:AWS_PAGER = ""

Write-Host "INFO: Checking endpoint '$Endpoint' in region '$Region'..."

# Comment: Check if the endpoint exists (returns "0" or "1").
$exists = aws sagemaker list-endpoints --region $Region --name-contains $Endpoint --query "length(Endpoints[?EndpointName=='$Endpoint'])" --output text 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "ERROR: AWS CLI error while listing endpoints. Check your credentials/region."
  aws sts get-caller-identity 2>$null | Out-Null
  exit 1
}

if ($exists -eq "0") {
  Write-Host "INFO: Endpoint '$Endpoint' not found. Nothing to delete."
  exit 0
}

Write-Host "INFO: Deleting endpoint '$Endpoint'..."
aws sagemaker delete-endpoint --region $Region --endpoint-name $Endpoint 2>$null
# Comment: If AWS returns a non-zero because it is already deleting, we will still poll below.

# Comment: Poll for disappearance (max ~2.5 minutes).
for ($i = 0; $i -lt 30; $i++) {
  Start-Sleep -Seconds 5
  $exists = aws sagemaker list-endpoints --region $Region --name-contains $Endpoint --query "length(Endpoints[?EndpointName=='$Endpoint'])" --output text 2>$null
  if ($exists -eq "0") { break }
}

if ($exists -eq "0") {
  Write-Host "SUCCESS: Deleted (or no longer present): $Endpoint"
  exit 0
} else {
  Write-Host "INFO: Still present; AWS may still be tearing down. Check the console if needed."
  exit 0
}
