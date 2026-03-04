# Rebuild baseline for DataQuality using a real capture sample (JSON Lines).
# Fix: use the correct constructor arg name 'role' (not 'role_arn').
# Also: use DatasetFormat helper to match capture format.

from sagemaker.model_monitor import (
    DatasetFormat,  # helper for dataset format
    DefaultModelMonitor,
)

region = "ca-central-1"  # keep your region
bucket = "mlops-bikeshare-387706002632-ca-central-1"  # your bucket
baseline_prefix = "monitoring/baseline/city=nyc"  # SAME prefix your JD reads from
role_arn = "arn:aws:iam::387706002632:role/mlops-bikeshare-sagemaker-exec"  # your exec role

# This file should be a SageMaker data-capture JSONL sample (one JSON object per line).
baseline_s3 = "s3://mlops-bikeshare-387706002632-ca-central-1/monitoring/baseline-inputs/baseline_flat.json"


monitor = DefaultModelMonitor(
    role=role_arn,  # pass the role ARN here
    instance_count=1,  # small is fine for baseline
    instance_type="ml.m5.large",
    volume_size_in_gb=20,
    max_runtime_in_seconds=1800,
)

monitor.suggest_baseline(
    baseline_dataset=baseline_s3,
    dataset_format=DatasetFormat.json(lines=True),  # 关键！改成扁平 JSONL
    output_s3_uri="s3://mlops-bikeshare-387706002632-ca-central-1/monitoring/baseline/city=nyc",
    wait=True,
    logs=True,
)

print("Baseline written under:", f"s3://{bucket}/{baseline_prefix}/")
