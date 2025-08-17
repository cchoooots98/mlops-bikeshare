# Operational SLOs and Alerting Thresholds

## Service Level Objectives (SLOs)

### Data Layer
- **Ingestion frequency**: Every 5 minutes.
- **Data lake latency**: ≤ 3 minutes (95th percentile).
- **Validation pass rate**: ≥ 99% of batches pass schema and quality checks.

### Model Layer
- **PR-AUC**: ≥ 0.70 on validation.
- **F1 (stockout class)**: ≥ 0.55.
- **Overfitting check**: Training–validation gap ≤ 0.1.

### Service Layer
- **Endpoint P95 latency**: ≤ 200ms.
- **Inference batch success rate**: ≥ 99%.
- **Error rate**: < 1% per day.

### Monitoring Layer
- **Data drift (PSI)**: < 0.2 warning, < 0.3 critical.
- **Model quality (PR-AUC@24h, F1@24h)**: Degradation > 10% triggers alarm.
- **System availability**: ≥ 99.5% monthly uptime.
- **MTTA (Mean Time to Acknowledge)**: ≤ 10 minutes.

## Alerting Thresholds

| Category          | Metric                          | Warning                  | Critical                  |
|-------------------|---------------------------------|--------------------------|---------------------------|
| **Data Ingestion**| Latency (minutes)               | > 3                      | > 5                       |
|                   | Validation failure rate         | > 1% per hour            | > 5% per hour             |
| **Endpoint**      | P95 latency                     | > 200 ms                 | > 300 ms                  |
|                   | Invocation error rate           | > 0.5%                   | > 1%                      |
| **Model Drift**   | Population Stability Index (PSI)| > 0.2                    | > 0.3                     |
|                   | Feature missing rate            | > 1%                     | > 5%                      |
| **Model Quality** | PR-AUC@24h degradation          | > 5% vs. baseline        | > 10% vs. baseline        |
|                   | F1@24h degradation              | > 5% vs. baseline        | > 10% vs. baseline        |
| **System**        | Endpoint availability           | < 99.5% (monthly)        | < 99% (monthly)           |
|                   | Batch job failure rate          | > 1%                     | > 5%                      |

## Error Budget
- **Data SLO budget**: ≤ 0.5% ingestion failures per week.
- **Model SLO budget**: ≤ 1 day per month below target PR-AUC/F1.
- **Service SLO budget**: ≤ 0.5% downtime/unavailability per month.

## Escalation
- **Critical alerts**: Notify on-call via Slack + Email (24/7).
- **Warning alerts**: Logged in CloudWatch/Grafana dashboards, reviewed during daily standup.
- **Runbook**: See `docs/monitoring_runbook.md` for troubleshooting steps and bypass strategies.
