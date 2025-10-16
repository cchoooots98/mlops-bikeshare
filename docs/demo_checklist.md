# Bikeshare MLOps — Demo Checklist (Windows + PowerShell)

**Purpose**: A step‑by‑step script to (1) run a low‑cost environment during normal days and (2) bring the full demo online in **~7 minutes** for interviews.  
**Audience**: You (and any reviewer) running on Windows, VS Code, PowerShell, AWS CLI v2.

---

## 0) warm‑up on demo

1. **Set env + health check**
   ```powershell
   # Comment: Set profile/region/endpoint shown in the dashboard.
   $env:AWS_PROFILE="Shirley"
   $env:AWS_REGION="ca-central-1"
   $env:SM_ENDPOINT="bikeshare-prod"

   # Comment: Quick sanity checks
   aws sts get-caller-identity
   $env:AWS_REGION
   $env:SM_ENDPOINT
   ```

2. **Start the endpoint** (waits for InService + publishes a heartbeat)
   ```powershell
   .\scripts\start_demo.ps1
   ```

3. **(Optional) Resume App Runner** (public URL for Streamlit)
   ```powershell
   aws apprunner resume-service --service-arn <YOUR_APP_RUNNER_ARN> --region $env:AWS_REGION
   ```

4. **(Optional) Re‑enable EventBridge rules** (live ingest/infer/psi)
   ```powershell
   $names = aws events list-rules --region $env:AWS_REGION --name-prefix bikeshare- --query 'Rules[].Name' --output json | ConvertFrom-Json
   $names | % { if ($_ -and $_.Trim()) { aws events enable-rule --region $env:AWS_REGION --name $_ | Out-Null } }
   ```

5. **Open the dashboard**
   ```powershell
   streamlit run app/dashboard.py
   # Local:  http://localhost:8501
   # Public (if App Runner): https://<random>.awsapprunner.com
   ```

6. **Force a fresh KPI push** (if charts look empty)
   - Manually “Run workflow” → `.github/workflows/publish_metrics.yml` on GitHub, or run your local publisher.
   - Expect to see **PR-AUC‑24h / F1‑24h / ThresholdHitRate / Samples** update within 1–2 minutes.

7. **Smoke‑invoke** (proves endpoint works)
   ```powershell
   python test/smoke_invoke.py --endpoint-name $env:SM_ENDPOINT
   ```

---

## 1) One‑time prerequisites

- **Installed**: AWS CLI v2, Python 3.11, Git, Docker Desktop (for App Runner build), VS Code.
- **AWS credentials**: profile `Shirley` with permissions to SageMaker, CloudWatch, Events, Lambda, ECR, App Runner, S3.
- **Repo env**:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```
- **Central config file**: `src/utils/config.py` must contain:
  ```python
  REGION="ca-central-1"; CITY="nyc"; SM_ENDPOINT="bikeshare-prod"; CW_NAMESPACE="Bikeshare/Model"
  ```

---

## 2) Cost‑OFF mode (daily idle)

Use when **not** interviewing.

### 2.1 Delete the endpoint (stop per‑hour charges)
```powershell
.\scripts\stop_demo.ps1
```

### 2.2 Pause App Runner (if used)
```powershell
aws apprunner pause-service --service-arn <YOUR_APP_RUNNER_ARN> --region $env:AWS_REGION
```

### 2.3 Disable EventBridge rules (ingest/infer/psi)
```powershell
$REGION = $env:AWS_REGION; if (-not $REGION) { $REGION = "ca-central-1" }
$env:AWS_PAGER = ""
$names = aws events list-rules --region $REGION --name-prefix bikeshare- --query 'Rules[].Name' --output json | ConvertFrom-Json
$names | % { if ($_ -and $_.Trim()) { Write-Host "Disabling $_ ..."; aws events disable-rule --region $REGION --name $_ | Out-Null } }
aws events list-rules --region $REGION --name-prefix bikeshare- --query 'Rules[].{Name:Name,State:State}' --output table
```

### 2.4 (Optional) Block Lambda via reserved concurrency = 0
```powershell
$funcs = @("bikeshare-infer","bikeshare-publish-psi","mlops-bikeshare-gbfs-ingest","mlops-bikeshare-weather-ingest","bikeshare-partition-repair")
foreach ($f in $funcs) { aws lambda put-function-concurrency --function-name $f --reserved-concurrent-executions 0 2>$null | Out-Null }
```

### 2.5 Logs retention = 7 days (cheap)
```powershell
$groups = @(
  "/aws/sagemaker/Endpoints/bikeshare-prod",
  "/aws/lambda/bikeshare-infer",
  "/aws/lambda/bikeshare-publish-psi",
  "/aws/lambda/mlops-bikeshare-gbfs-ingest",
  "/aws/lambda/mlops-bikeshare-weather-ingest",
  "/aws/lambda/bikeshare-partition-repair"
)
foreach ($g in $groups) { aws logs put-retention-policy --log-group-name $g --retention-in-days 7 2>$null | Out-Null }
```

### 2.6 (Optional) Disable or delete alarms
```powershell
$ALRMS = aws cloudwatch describe-alarms --alarm-name-prefix "bikeshare-" --query "MetricAlarms[].AlarmName" --output json | ConvertFrom-Json
foreach ($a in $ALRMS) { aws cloudwatch disable-alarm-actions --alarm-names $a | Out-Null }
# Cheapest option:
# aws cloudwatch delete-alarms --alarm-names $($ALRMS -join ' ')
```

### 2.7 (Optional) ECR lifecycle policy
```powershell
$REPO="bikeshare-app"
$POLICY='{"rules":[{"rulePriority":1,"description":"keep last 2 images","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":2},"action":{"type":"expire"}}]}'
aws ecr put-lifecycle-policy --repository-name $REPO --lifecycle-policy-text $POLICY
```

---

## 3) Interview mode (bring‑up)

### 3.1 Re‑enable Lambda concurrency (if you blocked it)
```powershell
$funcs = @("bikeshare-infer","bikeshare-publish-psi","mlops-bikeshare-gbfs-ingest","mlops-bikeshare-weather-ingest","bikeshare-partition-repair")
foreach ($f in $funcs) { aws lambda delete-function-concurrency --function-name $f 2>$null | Out-Null }
```

### 3.2 Re‑enable EventBridge
```powershell
$names = aws events list-rules --region $env:AWS_REGION --name-prefix bikeshare- --query 'Rules[].Name' --output json | ConvertFrom-Json
$names | % { if ($_ -and $_.Trim()) { aws events enable-rule --region $env:AWS_REGION --name $_ | Out-Null } }
```

### 3.3 Start endpoint (InService + warm heartbeat)
```powershell
.\scripts\start_demo.ps1
```

### 3.4 Resume App Runner (if used)
```powershell
aws apprunner resume-service --service-arn <YOUR_APP_RUNNER_ARN> --region $env:AWS_REGION
```

### 3.5 Kick metrics (if needed)
- Manually run GitHub Action **publish_metrics.yml**, or
- Run local publisher using `src/monitoring/metrics/metrics_helper.py` helpers.

### 3.6 Open tabs to present
1. **docs/architecture.md** — one diagram + bullet pipeline.
2. **docs/ops_sla.md** — SLOs + alarms + rollback path.
3. **docs/monitoring_runbook.md** — triage steps.
4. **Streamlit dashboard** — Map + KPIs (PR‑AUC‑24h / F1‑24h / Samples / PSI / Heartbeat).
5. **GitHub Actions** — `promote_prod.yml` (admission gate + rollback).
6. **Terminal** — ready for `python test/smoke_invoke.py --endpoint-name $env:SM_ENDPOINT`.

---

## 4) 90‑second narration (what to say)

- **Ingestion** via Lambda (GBFS + weather) → **S3** → partition repair.  
- **Features/Labels** in S3 + Athena; training via MLflow/SageMaker; model registered.  
- **Deployment** with blue/green via `promote_prod.yml`; **admission gate** uses 24h KPIs.  
- **Monitoring**: custom metrics (PR‑AUC/F1/HitRate/Samples/Heartbeat、PSI) to CloudWatch; alarms trigger rollback.  
- **Cost guardrails**: endpoint deletion, paused schedules, log retention, ECR lifecycle.  

---

## 5) Post‑demo teardown (2 minutes)

```powershell
# Comment: Stop compute
.\scripts\stop_demo.ps1
aws apprunner pause-service --service-arn <YOUR_APP_RUNNER_ARN> --region $env:AWS_REGION

# Comment: Disable schedules
$names = aws events list-rules --region $env:AWS_REGION --name-prefix bikeshare- --query 'Rules[].Name' --output json | ConvertFrom-Json
$names | % { if ($_ -and $_.Trim()) { aws events disable-rule --region $env:AWS_REGION --name $_ | Out-Null } }
```

---

## 6) Quick troubleshooting

- **No metrics on charts** → Run `publish_metrics.yml` once; confirm namespace `Bikeshare/Model` and dimensions `{EndpointName=bikeshare-prod, City=nyc}`.  
- **Endpoint not found** → `aws sagemaker list-endpoints --name-contains bikeshare --query 'Endpoints[].{Name:EndpointName,Status:EndpointStatus}' --output table`  
- **Throttling on CloudWatch** → helper auto‑retries; otherwise reduce frequency to 10–15 min.  
- **Lambda still firing** → check EventBridge rule state + reserved concurrency (0 blocks all).  
- **Dashboard 403 (App Runner)** → confirm service **resumed** and env vars (CITY/SM_ENDPOINT) are set.


