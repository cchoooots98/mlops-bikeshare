import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path
from typing import Sequence

import boto3
import botocore
from botocore.exceptions import ClientError

from src.model_package import (
    build_deployment_state,
    load_package_manifest,
    write_deployment_state,
)


def ts_suffix() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy/update a SageMaker endpoint from a model package.")
    parser.add_argument("--endpoint-name", required=True, help="Endpoint name, e.g. bikeshare-bikes-staging")
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--image-uri", required=True, help="Inference container image URI")
    parser.add_argument("--package-s3-uri", default=None, help="S3 URI to a packaged model tar.gz")
    parser.add_argument("--model-data", default=None, help="Deprecated alias of --package-s3-uri")
    parser.add_argument("--package-dir", default=None, help="Optional local model package directory for deployment metadata.")
    parser.add_argument("--instance-type", required=True, help="Instance type, e.g. ml.m5.large")
    parser.add_argument("--region", required=True, help="AWS region, e.g. eu-west-3")
    parser.add_argument("--environment", default="staging", help="Deployment environment label.")
    parser.add_argument("--deployment-state-path", default=None, help="Optional local deployment state JSON path.")
    return parser.parse_args(argv)


def assert_image_region_matches(image_uri: str, region: str) -> None:
    match = re.search(r"\.ecr\.([a-z0-9-]+)\.amazonaws\.com/", image_uri)
    if not match:
        print(f"[WARN] Could not parse region from ECR image URI: {image_uri}")
        return
    image_region = match.group(1)
    if image_region != region:
        raise ValueError(f"ECR image is in {image_region}, but --region is {region}. They must match.")


def create_model(sm, model_name: str, image_uri: str, package_s3_uri: str, exec_role_arn: str) -> None:
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={"Image": image_uri, "ModelDataUrl": package_s3_uri, "Environment": {}},
        ExecutionRoleArn=exec_role_arn,
    )


def create_endpoint_config(sm, endpoint_config_name: str, model_name: str, instance_type: str) -> None:
    sm.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": instance_type,
                "ContainerStartupHealthCheckTimeoutInSeconds": 600,
            }
        ],
    )


def wait_until_not_in_progress(sm, endpoint_name: str, timeout_sec: int = 900, poll_sec: int = 15) -> str:
    progress = {"Creating", "Updating", "SystemUpdating", "RollingBack", "Deleting"}
    deadline = time.time() + timeout_sec
    while True:
        try:
            desc = sm.describe_endpoint(EndpointName=endpoint_name)
            status = desc.get("EndpointStatus", "Unknown")
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ValidationException":
                return "NonExistent"
            raise
        if status not in progress:
            return status
        if time.time() >= deadline:
            raise TimeoutError(f"Endpoint '{endpoint_name}' stuck in {status}")
        time.sleep(poll_sec)


def upsert_endpoint(sm, endpoint_name: str, endpoint_config_name: str) -> None:
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        wait_until_not_in_progress(sm, endpoint_name)
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)
    except ClientError as exc:
        message = exc.response.get("Error", {}).get("Message", "")
        if exc.response.get("Error", {}).get("Code") == "ValidationException" and "Could not find endpoint" in message:
            sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)
            return
        raise


def wait_in_service(sm, endpoint_name: str) -> None:
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)


def maybe_write_deployment_state(
    *,
    package_dir: str | None,
    environment: str,
    deployment_state_path: str | None,
    endpoint_name: str,
) -> str | None:
    if not package_dir or not deployment_state_path:
        return None
    manifest = load_package_manifest(package_dir)
    state = build_deployment_state(
        package_dir,
        manifest,
        environment=environment,
        source="sagemaker_deploy",
        endpoint_name=endpoint_name,
    )
    return write_deployment_state(deployment_state_path, state)


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    package_s3_uri = args.package_s3_uri or args.model_data
    if not package_s3_uri:
        raise ValueError("provide --package-s3-uri (or deprecated --model-data)")
    assert_image_region_matches(args.image_uri, args.region)

    sm = boto3.client("sagemaker", region_name=args.region)
    suffix = ts_suffix()
    model_name = f"{args.endpoint_name}-model-{suffix}"
    endpoint_config_name = f"{args.endpoint_name}-config-{suffix}"

    create_model(sm, model_name, args.image_uri, package_s3_uri, args.role_arn)
    create_endpoint_config(sm, endpoint_config_name, model_name, args.instance_type)
    wait_until_not_in_progress(sm, args.endpoint_name)
    upsert_endpoint(sm, args.endpoint_name, endpoint_config_name)
    wait_in_service(sm, args.endpoint_name)
    state_path = maybe_write_deployment_state(
        package_dir=args.package_dir,
        environment=args.environment,
        deployment_state_path=args.deployment_state_path,
        endpoint_name=args.endpoint_name,
    )
    result = {
        "endpoint_name": args.endpoint_name,
        "environment": args.environment,
        "endpoint_config_name": endpoint_config_name,
        "model_name": model_name,
        "package_s3_uri": package_s3_uri,
        "deployment_state_path": state_path,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FATAL] Deployment failed: {exc}")
        sys.exit(1)
