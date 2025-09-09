# pipelines/deploy_staging.py
# Purpose:
#   Deploy an MLflow-registered model to AWS SageMaker as a real-time endpoint.
#   Supports selecting by stage (e.g., Staging) OR by explicit version number.
#   Adds optional --image-url to pin a specific ECR image (Docker v2 manifest).
#
# How it works:
#   1) Resolve model_uri from MLflow Model Registry: models:/<name>/<stage|version>
#   2) Call mlflow.sagemaker.deploy(...) to create/update the endpoint.
#   3) Print the tracking URI and endpoint info for debugging.
#
# Notes:
#   - Make sure the same MLflow tracking DB is used everywhere (training, register, deploy).
#   - On Windows + PowerShell, set an absolute SQLite URI to avoid path confusion:
#       $env:MLFLOW_TRACKING_URI = "sqlite:///E:/算法自学/End2EndProject/mlops-bikeshare/mlflow.db"
#   - If you pass --image-url, it must point to an ECR image with Docker v2 manifest
#     (SageMaker does not accept OCI manifest lists).

import argparse
import os

import mlflow


def main():
    parser = argparse.ArgumentParser(description="Deploy MLflow model to SageMaker endpoint.")
    parser.add_argument("--model-name", required=True, help="MLflow Registry name, e.g., 'bikeshare_risk'.")
    parser.add_argument(
        "--stage", default=None, help="Stage to deploy from (e.g., 'Staging'). Mutually exclusive with --version."
    )
    parser.add_argument(
        "--version", default=None, type=str, help="Specific model version to deploy (mutually exclusive with --stage)."
    )
    parser.add_argument(
        "--endpoint-name", required=True, help="SageMaker Endpoint name to create/update, e.g., 'bikeshare-staging'."
    )
    parser.add_argument("--role-arn", required=True, help="IAM execution role ARN for SageMaker.")
    parser.add_argument("--instance-type", default="ml.m5.large", help="Instance type (cost/perf tradeoff).")
    parser.add_argument("--region", default="ca-central-1", help="AWS region, e.g., 'ca-central-1'.")
    parser.add_argument("--timeout", default=600, type=int, help="Timeout in seconds for deployment.")
    parser.add_argument(
        "--image-url",
        default=None,
        help="Optional ECR image URI to pin, e.g., "
        "'387706002632.dkr.ecr.ca-central-1.amazonaws.com/mlflow-pyfunc:3.3.2'.",
    )
    args = parser.parse_args()

    # Ensure a consistent MLflow tracking URI (absolute path recommended on Windows).
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", "sqlite:///E:/算法自学/End2EndProject/mlops-bikeshare/mlflow.db"
    )
    mlflow.set_tracking_uri(tracking_uri)
    print("Using MLFLOW_TRACKING_URI:", tracking_uri)

    # Resolve models:/ URI from either stage or version.
    if args.stage and args.version:
        raise ValueError("Use either --stage or --version, not both.")
    if args.stage:
        model_uri = f"models:/{args.model_name}/{args.stage}"
    elif args.version:
        model_uri = f"models:/{args.model_name}/{args.version}"
    else:
        raise ValueError("You must provide --stage or --version.")

    print("Deploying model_uri:", model_uri)
    print("Endpoint name:", args.endpoint_name)
    print("Region:", args.region)
    print("Instance type:", args.instance_type)
    if args.image_url:
        print("Pinning image_url:", args.image_url)

    # Call MLflow's SageMaker deploy.
    # If image_url is provided, pass it through so SageMaker uses your known-good Docker v2 image.
    mlflow.sagemaker.deploy(
        app_name=args.endpoint_name,  # Endpoint name
        model_uri=model_uri,  # models:/... URI
        mode="create",  # Use "replace" to tear down and recreate
        region_name=args.region,
        execution_role_arn=args.role_arn,
        instance_type=args.instance_type,
        timeout_seconds=args.timeout,
        image_url=args.image_url,  # <-- NEW: optional override
    )

    print(f"Deployed {model_uri} to SageMaker endpoint: {args.endpoint_name}")


if __name__ == "__main__":
    main()
