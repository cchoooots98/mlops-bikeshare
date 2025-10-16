# Operations SLO/SLA and Runbook 

_Last updated: 2025-10-14 • Region: **ca-central-1** • City: **nyc**_

This document defines service targets, alert thresholds, admission gates for promotion to **prod**, rollback policy, owners, and reporting cadence.

---

## 1) SLOs and SLIs

| Area | SLI (namespace • metric • dims) | Target | Notes |
|---|---|---|---|
| **Endpoint latency p95** | AWS/SageMaker • `ModelLatency` (5-min bins) • `{EndpointName, VariantName=AllTraffic}` | **Target ≤ 200 ms**, **SLO ≤ 300 ms** | Prod alarms use 200/300 ms thresholds. Unit: microseconds. |
| **5xx error count** | AWS/SageMaker • `Invocation5XXErrors` | **0** | Any 5xx is critical. |
| **4xx error rate** | Math: `Invocation4XXErrors / Invocations` | **< 1%** | Warn only (operator action; not always model fault). |
| **Model quality – PR-AUC (24h)** | Bikeshare/Model • `PR-AUC-24h` • `{EndpointName, City}` | **≥ 0.70 target** | Current prod warn is absolute; see §2. |
| **Model quality – F1 (24h)** | Bikeshare/Model • `F1-24h` • `{EndpointName, City}` | **≥ 0.55 target** | Current prod warn is absolute; see §2. |
| **Feature drift (PSI)** | Bikeshare/Model • `PSI` • `{EndpointName, City}` | **Warn ≥ 0.20, Crit ≥ 0.30** | Hourly. |
| **Prediction cadence (heartbeat)** | Bikeshare/Model • `PredictionHeartbeat` | **≥ 144 per 24h** | 10-min cadence × 24h. |

---

## 2) Alert thresholds (warning vs critical) — **prod**

These names match your CloudWatch alarms.

| Group | Condition | Level | Alarm name | Period / Evaluation |
|---|---|---|---|---|
| Latency p95 | `> 200 ms` | Warn | `prod-endpoint-latency-p95-warn` | 5-min · 3/3 |
| Latency p95 | `> 300 ms` | Crit | `prod-endpoint-latency-p95-crit` | 5-min · 5/5 |
| 5xx errors | `> 0` | Crit | `prod-endpoint-5xx-crit` | 5-min · 1/1 |
| 4xx rate | `> 1%` | Warn | `prod-endpoint-4xx-warn` | 5-min · 2/2 |
| Latency anomaly | Outside band (width 2.5) | Warn | `prod-latency-anomaly` | 5-min · 3/3 |
| Low traffic (24h) | `Invocations ≤ 30/h` | Warn | `prod-low-traffic-24h` | 1-h · 24/24 |
| Short-window zero | `Invocations == 0` | Crit | `prod-20m-zero` | 5-min · 4/4 |
| Invoke-zero composite | Children in ALARM | Crit | `prod-endpoint-invoke-zero` | Composite (short-window zero + low-traffic) |
| PR-AUC (24h) | `< 0.851` | Warn | `prod-quality-prauc-24h-warn` | 30-min · 3/3 |
| F1 (24h) | `< 0.753` | Warn | `prod-quality-f1-24h-warn` | 30-min · 3/3 |
| PSI | `≥ 0.20` (warn) / `≥ 0.30` (crit) | Warn/Crit | `prod-feature-drift-warn` / `prod-feature-drift-crit` | 1-h · 1/1 |

> **Note:** PR-AUC/F1 absolute thresholds are pinned to the current baseline. Refresh them when a new baseline is approved (see §4).

---

## 3) Error budget and burn policy

- **Availability SLO:** 99.5% monthly for inference (p95 latency SLO + 5xx=0).  
- **Fast burn:** if 5xx>0 **or** p95>300 ms for ≥15 min, page immediately and start rollback (§8).  
- **Slow burn:** if PR-AUC/F1 warn persists >2 h, create incident, restrict traffic if needed, schedule model triage.

---

## 4) Baselines, degradations, promotions

- Baseline values (PR-AUC, F1, PSI) are stored with the model version (registry notes).  
- When promoting a new model, **synchronize** alarm thresholds to the refreshed baseline.  
- Degradation requires: incident doc, RCA, action items, and—if applicable—baseline refresh PR.

---

## 5) Ownership, escalation, evidence

- **Owner:** MLOps (primary) and Data Science (secondary).  
- **Escalation path:** On-call → Tech Lead → Product.  
- **Evidence:** CloudWatch dashboard screenshots, alarm timelines, Athena queries, and model evaluation logs.

---

## 6) Current targets vs observed

See the “Business Dashboard” and “System Health” sections for the last 24–72 h.

---

## 7) Reporting cadence

- Weekly: quality and drift review.  
- Monthly: cost vs budget, SLO compliance, incident summary.

---

## 8) Prod admission gate & rollback (Step-10)

**Admission gate (pre-promote, 24 h window):**
1. `PR-AUC-24h ≥ 0.70` and `F1-24h ≥ 0.55` (Bikeshare/Model, dims `{EndpointName, City}`).
2. `ModelLatency p95 ≤ 200 ms` and `Invocation5XXErrors = 0` (AWS/SageMaker).
3. `PredictionHeartbeat ≥ 144` (10-min batches × 24h).
4. `PSI < 0.20` (if ≥0.20, require explicit waiver).

**Implementation:** run `tools/check_gate.py` in CI. If it fails, promotion is blocked.

**Rollback:**
- **A/B:** `UpdateEndpointWeightsAndCapacities` to restore `Baseline=1.0, Candidate=0.0`.  
- **Blue/green:** `UpdateEndpoint` back to the previous EndpointConfig.  
- Create an incident record and attach graphs; refresh baselines if the new model is rejected.

---

## 9) Cost guardrails

- Stop or downsize **staging** after prod is stable.  
- Emit custom metrics per **batch** (not per request); keep **5-min** periods.  
- S3 lifecycle for `datacapture/`, `monitoring/`, `athena_results/`.  
- Athena: partitions + compression to reduce scanned bytes.
