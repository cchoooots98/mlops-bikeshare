# Operational SLOs & Alerting — mlops-bikeshare

_Last updated: 2025-10-02 • Region: **ca-central-1** • City: **nyc**_  
_Endpoints: **bikeshare-staging**, **bikeshare-prod**_  
_Routing: SNS **sm-alerts** → Slack + Email (Step-8 drill verified; MTTA = **2.6 min** on 2025-10-02)_

This document defines **SLOs** (targets), **SLIs** (how we measure), and **alert thresholds** (when to page). It aligns with the Step-8 acceptance criteria:
- **Drift/Quality/System** alerts can trigger and route (verified by forced-ALARM drill).
- Dashboard shows **24–72h** trends (quality, latency, errors, heartbeat).
- **MTTA ≤ 10 min** (drill measured **2.6 min**).

---

## 1) Service Level Objectives (SLOs) & SLIs

> **Windows:** Unless specified, SLOs are evaluated over a **rolling 30-day** window.  
> **Dashboards:** The operations dashboard exposes **1d/3d/1w** (24–72h) trend views used for on-call triage and weekly review.

### 1.1 Data Layer
| SLO | SLI (measurement) | Target | Notes |
|---|---|---|---|
| **Ingestion frequency** | New raw batch (GBFS/weather) every **5 min** | **≥ 95%** on-time | Measured by custom freshness metric / partitions. |
| **Data lake latency (p95)** | Time from source fetch → S3/Glue partition visible | **≤ 3 min** | Athena/Glue partition freshness checks. |
| **Validation pass rate** | % batches passing schema/quality checks | **≥ 99%** | From ingestion Lambda logs/metrics. |

### 1.2 Model Layer
| SLO | SLI (measurement) | Target | Notes |
|---|---|---|---|
| **PR-AUC (staging gate)** | `PR-AUC-24h` (Bikeshare/Model) | **≥ 0.70** | Rolling 24h; used for promotion gate. |
| **F1 (stockout)** | `F1-24h` (Bikeshare/Model) | **≥ 0.55** | Rolling 24h at current threshold. |
| **Overfitting check** | Train–valid metric gap | **≤ 0.10** | Evaluated at training time; logged in MLflow. |

### 1.3 Service Layer (Online)
| SLO | SLI (measurement) | Target | Notes |
|---|---|---|---|
| **Endpoint availability (prod)** | 1 − `Invocation5XXErrors/Invocations` | **≥ 99.9%** | CloudWatch (AWS/SageMaker). Error budget ≈ 43m 49s / month. |
| **Endpoint latency p95** | `ModelLatency` p95 (5-min bins) | **≤ 300 ms** | Internal target **200 ms**; SLO 300 ms. |
| **Inference batch success** | Success ratio of scheduled runs | **≥ 99%** | Lambda/GitHub Actions success history. |
| **Daily error rate** | `InvocationErrorRate` | **< 1%** | 4xx + 5xx, excluding client errors if noisy. |

### 1.4 Monitoring Layer
| SLO | SLI (measurement) | Target | Notes |
|---|---|---|---|
| **Data drift (PSI)** | Feature PSI vs baseline | **Warn ≥ 0.2**, **Crit ≥ 0.3** | Baseline frozen at promotion; see §4.2. |
| **Model quality regression** | Δ(24h PR-AUC/F1) vs baseline | **Warn > 5%**, **Crit > 10%** | Baseline defined in §4.2. |
| **System availability** | Uptime of inference pipeline components | **≥ 99.5%** | Non-endpoint infra (Schedulers, Monitors). |
| **MTTA** | Alarm → first human view | **≤ 10 min** | Drill on 2025-10-02 = **2.6 min**. |
| **MTTR** | Alarm → full recovery | **≤ 60 min (Sev-2)**; **≤ 4 h (Sev-3)** | See runbook for bypass options. |

---

## 2) Alerting Thresholds (Warning vs Critical)

> **Philosophy:** SLOs represent steady-state targets. **Alerts** fire earlier to preserve the error budget.

| Category | Metric | Warning | Critical | Alarm Name(s) / Source |
|---|---|---:|---:|---|
| **Data Ingestion** | Freshness / latency | > **3 min** | > **5 min** | Custom freshness alarm (Athena/Glue) |
|  | Validation failure rate | > **1% / h** | > **5% / h** | Ingestion Lambda metric |
| **Endpoint** | p95 latency | > **800 ms** | > **1.5 s** | `sm-prod-avg-latency`, `sm-prod-latency` (AWS/SageMaker) |
|  | Invocation error rate | > **0.5%** | > **1%** | `sm-prod-5XX` (+4XX if desired) |
| **Model Drift** | PSI | ≥ **0.2** | ≥ **0.3** | (Add feature-level PSI alarms as needed) |
|  | Feature missing rate | > **1%** | > **5%** | Data-quality monitor or custom metric |
| **Model Quality** | PR-AUC@24h drop | > **5%** vs baseline | > **10%** vs baseline | `staging-prauc-low` (recommended) |
|  | F1@24h drop | > **5%** vs baseline | > **10%** vs baseline | `staging-f1-low` (exists) |
| **System** | Endpoint availability (monthly) | < **99.5%** | < **99%** | Composite or daily roll-up |
|  | Batch job failure rate | > **1%** | > **5%** | `bikeshare-infer-Errors-gt0`, duration alarms |
| **Monitors** | Drift/quality job status | — | **Failed** | `bikeshare-data-drift-failed`, `bikeshare-data-quality-failed` |

> **Note:** Your Step-8 drill also validated routing for all above via SNS **sm-alerts**.

---

## 3) Error Budgets & Burn Policy

### 3.1 Budgets
- **Endpoint availability 99.9% (monthly)** → **43m 49s** of downtime.  
- **System availability 99.5% (monthly)** → **3h 39m** of downtime.

### 3.2 Burn policy
- **> 25%** budget consumed in a week → pause risky changes; prioritize reliability fixes.  
- **> 50%** consumed → freeze promotions; only mitigations/rollbacks allowed.  
- **> 100%** consumed → post-mortem + prevention plan; track to closure.

---

## 4) Baselines, Degradations & Promotions

### 4.1 Quality SLI windows
- **24h rolling** metrics are used for PR-AUC/F1 alarms.  
- **Samples-24h** and **ThresholdHitRate-24h** provide context for low-sample or threshold shifts.

### 4.2 Baseline definition
- For each model **promotion**, capture a **baseline** from the **previous 7–14 days** of stable prod/staging runs.  
- Baseline values (PR-AUC, F1, PSI distributions) are stored with the model version (MLflow/registry notes).  
- **Degradation** is computed as relative drop vs that frozen baseline (e.g., PR-AUC↓ > 10%).

---

## 5) Ownership, Escalation & Evidence

### 5.1 On-call & escalation
- **Critical alerts:** Page via Slack + Email (24/7). Primary: DS on duty; Secondary: MLOps.  
- **Warning alerts:** Visible in dashboards; reviewed in daily stand-up.  
- **Sev-1 impact:** Immediate rollback/traffic shift per runbook; notify stakeholders.

### 5.2 Step-8 evidence (kept for audits)
- **Routing proof:** SNS publish test → Slack/email delivered.  
- **Alarm drill:** Forced alarms across drift/quality/system on **2025-10-02**.  
- **MTTA:** **2.6 min** (ALARM → first human view).  
- **Dashboard:** 24–72h trends present for all critical signals.

### 5.3 Runbook
See **`docs/monitoring_runbook.md`** for troubleshooting trees, Athena/Glue queries, Lambda checks, and bypass/rollback commands.

---

## 6) Current Targets vs Observed (as of 2025-10-02)

- **Endpoint p95 latency:** Observed **~24–112 ms** (CloudWatch), **within SLO (≤ 300 ms)**.  
- **Model quality (staging):** 24h PR-AUC / F1 healthy and above gates.  
- **MTTA:** **2.6 min** drill (≤ 10 min SLO).  
- **Routing & alarms:** Verified end-to-end.

---

## 7) Reporting Cadence

- **Weekly:** SLO dashboard review (availability, latency p95, batch success, monitor status, drift/quality).  
- **Monthly:** Error-budget report, post-mortems, and threshold tuning; confirm baselines for any new model promotions.

