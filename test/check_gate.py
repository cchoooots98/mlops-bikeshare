# test/check_gate.py
# Purpose: Gate for target-specific promotion. Fetch last 24h metrics from CloudWatch and enforce admission thresholds.
# Run: python test/check_gate.py --endpoint bikeshare-bikes-staging --city paris --region eu-west-3 --target-name bikes --environment staging

import argparse
import math
import sys
from datetime import datetime, timedelta, timezone

import boto3

# ---- Admission thresholds (tune to your SLOs) ----
AUC_MIN = 0.70  # PR-AUC >= 0.70 over the last 24h (Average)
F1_MIN = 0.55  # F1 >= 0.55 over the last 24h (Average)
PSI_CORE_WARN = 0.20  # Core feature drift should stay below 0.20 over the last 24h.
P95_LATENCY_MAX_US = 200_000  # 200 ms in microseconds
FIVE_XX_MAX = 0  # No 5xx errors allowed over the window
PREDICTION_CADENCE_MINUTES = 15
OBSERVATION_WINDOW_HOURS = 24
HEARTBEAT_COVERAGE_MIN = 0.95
EXPECTED_HEARTBEATS = int((60 / PREDICTION_CADENCE_MINUTES) * OBSERVATION_WINDOW_HOURS)
# Allow minor scheduler jitter or a few missed batches, but still require strong continuity.
HEARTBEAT_MIN = math.ceil(EXPECTED_HEARTBEATS * HEARTBEAT_COVERAGE_MIN)  # 92/96 at 15-min cadence
MAX_HEARTBEAT_GAP_MINUTES = (PREDICTION_CADENCE_MINUTES * 2) + 5


def get_series(cw, namespace, metric, dims, minutes=24 * 60, stat="Average", period=300):
    """Query CloudWatch metric as a time series."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes)
    d = [{"Name": k, "Value": v} for k, v in dims.items()]
    q = [
        {
            "Id": "m1",
            "MetricStat": {
                "Metric": {"Namespace": namespace, "MetricName": metric, "Dimensions": d},
                "Period": period,
                "Stat": stat,
            },
            "ReturnData": True,
        }
    ]
    resp = cw.get_metric_data(StartTime=start, EndTime=end, MetricDataQueries=q, ScanBy="TimestampAscending")
    r = resp["MetricDataResults"][0]
    return list(zip(r.get("Timestamps", []), r.get("Values", [])))


def avg(values):
    return sum(values) / len(values) if values else None


def max_gap_minutes(points):
    timestamps = sorted(ts for ts, _ in points)
    if len(timestamps) < 2:
        return None
    gaps = [(current - previous).total_seconds() / 60 for previous, current in zip(timestamps, timestamps[1:])]
    return max(gaps) if gaps else None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, help="EndpointName, e.g., bikeshare-bikes-staging")
    ap.add_argument("--city", required=True, help="City dim for custom metrics, e.g., paris")
    ap.add_argument("--region", required=True, help="AWS region, e.g., eu-west-3")
    ap.add_argument("--target-name", required=True, choices=["bikes", "docks"])
    ap.add_argument(
        "--environment",
        required=True,
        choices=["staging", "production"],
        help="CloudWatch Environment dimension. Pass staging for staging endpoints and production for prod endpoints.",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    cw = boto3.client("cloudwatch", region_name=args.region)

    ns = "Bikeshare/Model"
    dims_full = {
        "Environment": args.environment,
        "EndpointName": args.endpoint,
        "City": args.city,
        "TargetName": args.target_name,
    }

    pr_auc = get_series(cw, ns, "PR-AUC-24h", dims_full, stat="Average")
    f1 = get_series(cw, ns, "F1-24h", dims_full, stat="Average")
    psi_core = get_series(cw, ns, "PSI_core", dims_full, stat="Average")

    pr_auc_avg = avg([v for _, v in pr_auc])
    f1_avg = avg([v for _, v in f1])
    psi_core_max = max([v for _, v in psi_core]) if psi_core else 0.0

    sm_ns = "AWS/SageMaker"
    dims_sm = {"EndpointName": args.endpoint, "VariantName": "AllTraffic"}

    p95_latency = get_series(cw, sm_ns, "ModelLatency", dims_sm, stat="p95")
    p95_max = max([v for _, v in p95_latency]) if p95_latency else None

    five_xx = get_series(cw, sm_ns, "Invocation5XXErrors", dims_sm, stat="Sum")
    five_xx_sum = sum([v for _, v in five_xx]) if five_xx else 0

    hb = get_series(cw, ns, "PredictionHeartbeat", dims_full, stat="Sum")
    hb_sum = sum([v for _, v in hb]) if hb else 0
    hb_max_gap = max_gap_minutes(hb)

    failures = []

    if pr_auc_avg is None or pr_auc_avg < AUC_MIN:
        failures.append(f"PR-AUC-24h avg {pr_auc_avg} < {AUC_MIN}")
    if f1_avg is None or f1_avg < F1_MIN:
        failures.append(f"F1-24h avg {f1_avg} < {F1_MIN}")

    if p95_max is None or p95_max > P95_LATENCY_MAX_US:
        failures.append(f"ModelLatency p95 max {p95_max} us > {P95_LATENCY_MAX_US} us")

    if five_xx_sum > FIVE_XX_MAX:
        failures.append(f"Invocation5XXErrors sum {five_xx_sum} > {FIVE_XX_MAX}")

    if hb_sum < HEARTBEAT_MIN:
        failures.append(
            f"PredictionHeartbeat sum {hb_sum} < {HEARTBEAT_MIN} "
            f"({HEARTBEAT_COVERAGE_MIN:.0%} of expected {EXPECTED_HEARTBEATS} batches in 24h)"
        )
    if hb_max_gap is None or hb_max_gap > MAX_HEARTBEAT_GAP_MINUTES:
        failures.append(f"PredictionHeartbeat max gap {hb_max_gap} min > {MAX_HEARTBEAT_GAP_MINUTES} min")

    if psi_core_max >= PSI_CORE_WARN:
        failures.append(f"PSI_core max {psi_core_max} >= {PSI_CORE_WARN} (core feature drift)")

    if failures:
        print("Admission gate FAILED:")
        for failure in failures:
            print(" -", failure)
        sys.exit(1)

    print("Admission gate PASSED: all thresholds satisfied for the last 24h.")


if __name__ == "__main__":
    main()
