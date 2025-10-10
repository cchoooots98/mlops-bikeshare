# test/check_gate.py
# Purpose: Gate for Prod promotion. Fetch last 24h metrics from CloudWatch and enforce admission thresholds.
# Run: python test/check_gate.py --endpoint bikeshare-staging --city nyc --region ca-central-1

import argparse
import sys
from datetime import datetime, timedelta, timezone
import boto3

# ---- Admission thresholds (tune to your SLOs) ----
AUC_MIN = 0.70           # PR-AUC >= 0.70 over the last 24h (Average)
F1_MIN = 0.55            # F1 >= 0.55 over the last 24h (Average)
PSI_WARN = 0.20          # PSI should stay below 0.20 (optional gate; warn/fail as you prefer)
P95_LATENCY_MAX_US = 200_000   # 200 ms in microseconds (SageMaker ModelLatency unit is µs)
FIVE_XX_MAX = 0          # No 5xx errors allowed over the window
HEARTBEAT_MIN = 6*10    # Expect ~6 batches/hour (10-min cadence) => 144 in 24h

def get_series(cw, namespace, metric, dims, minutes=24*60, stat="Average", period=300):
    """Query CloudWatch metric as a time series."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes)
    d = [{"Name": k, "Value": v} for k, v in dims.items()]
    q = [{
        "Id": "m1",
        "MetricStat": {
            "Metric": {"Namespace": namespace, "MetricName": metric, "Dimensions": d},
            "Period": period,  # 300s matches your 5-min cadence and lowers API cost
            "Stat": stat
        },
        "ReturnData": True
    }]
    resp = cw.get_metric_data(StartTime=start, EndTime=end, MetricDataQueries=q, ScanBy="TimestampAscending")
    r = resp["MetricDataResults"][0]
    return list(zip(r.get("Timestamps", []), r.get("Values", [])))

def avg(values):
    return sum(values)/len(values) if values else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, help="EndpointName, e.g., bikeshare-staging")
    ap.add_argument("--city", required=True, help="City dim for custom metrics, e.g., nyc")
    ap.add_argument("--region", required=True, help="AWS region, e.g., ca-central-1")
    args = ap.parse_args()

    cw = boto3.client("cloudwatch", region_name=args.region)

    # --- Custom namespace KPIs from your Step 9/dashboard ---
    # Namespace and dimensions match your app/docs: Bikeshare/Model with {EndpointName, City}
    # Ref: architecture.md and dashboard.py
    ns = "Bikeshare/Model"
    dims_full = {"EndpointName": args.endpoint, "City": args.city}

    pr_auc = get_series(cw, ns, "PR-AUC-24h", dims_full, stat="Average")
    f1 = get_series(cw, ns, "F1-24h", dims_full, stat="Average")
    psi = get_series(cw, ns, "PSI", dims_full, stat="Average")  # optional gate

    pr_auc_avg = avg([v for _, v in pr_auc])
    f1_avg = avg([v for _, v in f1])
    psi_max = max([v for _, v in psi]) if psi else 0.0

    # --- SageMaker service metrics (AWS/SageMaker) ---
    sm_ns = "AWS/SageMaker"
    dims_sm = {"EndpointName": args.endpoint, "VariantName": "AllTraffic"}

    # Use true p95 for ModelLatency (your docs mention p95 SLO=200ms)
    p95_latency = get_series(cw, sm_ns, "ModelLatency", dims_sm, stat="p95")
    p95_max = max([v for _, v in p95_latency]) if p95_latency else None

    five_xx = get_series(cw, sm_ns, "Invocation5XXErrors", dims_sm, stat="Sum")
    five_xx_sum = sum([v for _, v in five_xx]) if five_xx else 0

    # Heartbeat (proxy for batch success & cadence)
    hb = get_series(cw, ns, "PredictionHeartbeat", dims_full, stat="Sum")
    hb_sum = sum([v for _, v in hb]) if hb else 0

    # ---- Evaluate gates ----
    failures = []

    if pr_auc_avg is None or pr_auc_avg < AUC_MIN:
        failures.append(f"PR-AUC-24h avg {pr_auc_avg} < {AUC_MIN}")
    if f1_avg is None or f1_avg < F1_MIN:
        failures.append(f"F1-24h avg {f1_avg} < {F1_MIN}")

    if p95_max is None or p95_max > P95_LATENCY_MAX_US:
        failures.append(f"ModelLatency p95 max {p95_max} µs > {P95_LATENCY_MAX_US} µs")

    if five_xx_sum > FIVE_XX_MAX:
        failures.append(f"Invocation5XXErrors sum {five_xx_sum} > {FIVE_XX_MAX}")

    if hb_sum < HEARTBEAT_MIN:
        failures.append(f"PredictionHeartbeat sum {hb_sum} < {HEARTBEAT_MIN} (batches in 24h)")

    # Optional: treat PSI>0.2 as warning or failure
    if psi_max >= PSI_WARN:
        failures.append(f"PSI max {psi_max} ≥ {PSI_WARN} (feature drift)")

    if failures:
        print("❌ Admission gate FAILED:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    else:
        print("✅ Admission gate PASSED: all thresholds satisfied for the last 24h.")

if __name__ == "__main__":
    main()
