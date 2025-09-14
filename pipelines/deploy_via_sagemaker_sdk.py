# pipelines/deploy_via_sagemaker_sdk.py
# Purpose: Create/Update a SageMaker endpoint with robust logging.
# Works with a BYOC image (e.g., mlflow-pyfunc) + a model tar in S3.
# All steps are commented in English.

import argparse
import datetime as dt
import re
import sys

import boto3
import botocore
import time
from botocore.exceptions import ClientError


def ts_suffix() -> str:
    """Return a UTC timestamp suffix to make unique names on every deploy."""
    return dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def parse_args():
    """Parse CLI arguments passed from your PowerShell command."""
    p = argparse.ArgumentParser(description="Deploy/Update a SageMaker endpoint")
    p.add_argument("--endpoint-name", required=True, help="Endpoint name, e.g., bikeshare-staging")
    p.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    p.add_argument("--image-uri", required=True, help="ECR image URI for inference container")
    p.add_argument("--model-data", required=True, help="S3 URI to model.tar.gz")
    p.add_argument("--instance-type", required=True, help="e.g., ml.m5.large")
    p.add_argument("--region", required=True, help="AWS region, e.g., ca-central-1")
    return p.parse_args()


def assert_image_region_matches(image_uri: str, region: str):
    """
    Basic guard: ECR image must be from the same region as the endpoint.
    Example image: 3877....dkr.ecr.ca-central-1.amazonaws.com/mlflow-pyfunc:3.3.2-v5
    """
    m = re.search(r"\.ecr\.([a-z0-9-]+)\.amazonaws\.com/", image_uri)
    if not m:
        print(f"[WARN] Could not parse region from ECR image URI: {image_uri}")
        return
    image_region = m.group(1)
    if image_region != region:
        raise ValueError(f"ECR image is in {image_region}, but --region is {region}. " "They must match.")


def create_model(sm, model_name: str, image_uri: str, model_data_s3: str, exec_role_arn: str):
    """
    Create a new SageMaker Model resource. We always create a fresh one
    (with a timestamp in the name) because Model resources are immutable.
    """
    print(f"[INFO] Creating Model: {model_name}")
    try:
        sm.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": image_uri,
                "ModelDataUrl": model_data_s3,
                # You can pass environment variables here if your container expects any
                "Environment": {},
            },
            ExecutionRoleArn=exec_role_arn,
        )
    except botocore.exceptions.ClientError as e:
        print("[ERROR] create_model failed:")
        print(e.response)
        raise


def create_endpoint_config(sm, endpoint_config_name: str, model_name: str, instance_type: str):
    """
    Create a new EndpointConfig with the provided Model.
    """
    print(f"[INFO] Creating EndpointConfig: {endpoint_config_name}")
    try:
        sm.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InitialInstanceCount": 1,
                    "InstanceType": instance_type,
                    # Increase startup timeout for slower images
                    "ContainerStartupHealthCheckTimeoutInSeconds": 600,
                }
            ],
        )
    except botocore.exceptions.ClientError as e:
        print("[ERROR] create_endpoint_config failed:")
        print(e.response)
        raise


def upsert_endpoint(sm, endpoint_name: str, endpoint_config_name: str):
    """
    Update the endpoint if it exists; otherwise create it.
    """
    try:
        # If describe works, endpoint exists -> update it
        sm.describe_endpoint(EndpointName=endpoint_name)
        print(f"[INFO] Updating Endpoint: {endpoint_name}")
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)
    except botocore.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        msg = e.response.get("Error", {}).get("Message", "")
        # Only create if the endpoint truly does not exist
        if code == "ValidationException" and "Could not find endpoint" in msg:
            print(f"[INFO] Creating Endpoint: {endpoint_name}")
            sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)
        else:
            print("[ERROR] update_endpoint failed; not creating because endpoint exists or another error occurred.")
            print(e.response)
            raise


def wait_in_service(sm, endpoint_name: str):
    """
    Wait for the endpoint to become InService. If it fails, print FailureReason.
    """
    print("[INFO] Waiting for endpoint to become InService ...")
    waiter = sm.get_waiter("endpoint_in_service")
    try:
        waiter.wait(EndpointName=endpoint_name)
        print("[OK] Endpoint InService")
    except botocore.exceptions.WaiterError:
        # When the waiter fails, we try to print the FailureReason
        print("[ERROR] WaiterError: endpoint did not become InService.")
        try:
            desc = sm.describe_endpoint(EndpointName=endpoint_name)
            status = desc.get("EndpointStatus", "Unknown")
            reason = desc.get("FailureReason", "N/A")
            print(f"[INFO] Current status: {status}")
            print(f"[INFO] FailureReason: {reason}")
        except botocore.exceptions.ClientError as e:
            # If we get ValidationException here, endpoint never got created
            print("[ERROR] describe_endpoint after waiter failure also failed:")
            print(e.response)
        # Re-raise to make the process non-zero exit for CI/CD visibility
        raise

def wait_until_not_in_progress(sm, endpoint_name: str, timeout_sec: int = 900, poll_sec: int = 15) -> str:
    """
    Wait until endpoint is NOT in a progress state like Creating/Updating/RollingBack/Deleting.
    Returns the final status (e.g., 'InService', 'Failed', 'OutOfService', or 'NonExistent').
    """
    progress = {"Creating", "Updating", "SystemUpdating", "RollingBack", "Deleting"}
    deadline = time.time() + timeout_sec
    while True:
        try:
            desc = sm.describe_endpoint(EndpointName=endpoint_name)
            status = desc.get("EndpointStatus", "Unknown")
        except ClientError as e:
            # If the endpoint truly does not exist, treat as ready to create
            if e.response.get("Error", {}).get("Code") == "ValidationException":
                return "NonExistent"
            raise
        if status not in progress:
            return status
        if time.time() >= deadline:
            raise TimeoutError(f"Endpoint '{endpoint_name}' stuck in {status}")
        time.sleep(poll_sec)

# --- modify upsert_endpoint(...) in-place ---
def upsert_endpoint(sm, endpoint_name: str, endpoint_config_name: str):
    """
    Update the endpoint if it exists; otherwise create it.
    Now robust against 'Cannot update in-progress endpoint'.
    """
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)  # exists -> we'll update
        print(f"[INFO] Updating Endpoint: {endpoint_name}")
        status = wait_until_not_in_progress(sm, endpoint_name)
        print(f"[INFO] Endpoint status before update: {status}")
        try:
            sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)
        except ClientError as e:
            msg = e.response.get("Error", {}).get("Message", "")
            if "Cannot update in-progress endpoint" in msg:
                print("[WARN] In progress again; waiting and retrying once ...")
                status = wait_until_not_in_progress(sm, endpoint_name)
                print(f"[INFO] Endpoint status before retry: {status}")
                sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)
            else:
                print("[ERROR] update_endpoint failed:")
                print(e.response)
                raise
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        msg = e.response.get("Error", {}).get("Message", "")
        if code == "ValidationException" and "Could not find endpoint" in msg:
            print(f"[INFO] Creating Endpoint: {endpoint_name}")
            sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)
        else:
            print("[ERROR] update_endpoint failed; not creating because endpoint exists or another error occurred.")
            print(e.response)
            raise

def main():
    args = parse_args()
    assert_image_region_matches(args.image_uri, args.region)

    # Create SDK clients
    sm = boto3.client("sagemaker", region_name=args.region)

    # Create unique names so we never collide with immutable resources
    suffix = ts_suffix()
    model_name = f"{args.endpoint_name}-model-{suffix}"
    endpoint_config_name = f"{args.endpoint_name}-config-{suffix}"

    # 1) Create Model
    create_model(
        sm=sm,
        model_name=model_name,
        image_uri=args.image_uri,
        model_data_s3=args.model_data,
        exec_role_arn=args.role_arn,
    )

    # 2) Create EndpointConfig
    create_endpoint_config(
        sm=sm, endpoint_config_name=endpoint_config_name, model_name=model_name, instance_type=args.instance_type
    )

    wait_until_not_in_progress(sm=sm, endpoint_name=args.endpoint_name)
    upsert_endpoint(sm, endpoint_name=args.endpoint_name, endpoint_config_name=endpoint_config_name)

    # 3) Update/Create Endpoint and wait
    upsert_endpoint(sm=sm, endpoint_name=args.endpoint_name, endpoint_config_name=endpoint_config_name)
    wait_in_service(sm, args.endpoint_name)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Ensure a non-zero exit code and print a readable error
        print(f"[FATAL] Deployment failed: {e}")
        sys.exit(1)
