# pipelines/deploy_via_sagemaker_sdk.py
# Purpose:
#   Create/Update a SageMaker real-time endpoint using a BYOC image (mlflow-pyfunc)
#   and a model.tar.gz you uploaded to S3. No 'import mlflow' anywhere.
#
# Notes:
#   - 'image_uri' must be a Docker v2 manifest image in your ECR (not OCI).
#   - 'model_data' is the S3 URI produced by the previous script.
#   - This script uses low-level boto3 for maximum control & compatibility.

import argparse
import time
import boto3

def wait_for_endpoint(sm_client, endpoint_name):
    # Poll endpoint status until InService or Failed
    while True:
        resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        print("EndpointStatus:", status)
        if status in ("InService", "Failed"):
            return status
        time.sleep(20)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint-name", required=True, help="SageMaker endpoint name, e.g., bikeshare-staging")
    ap.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    ap.add_argument("--image-uri", required=True, help="ECR image URI (Docker v2 manifest), e.g., .../mlflow-pyfunc:3.3.2")
    ap.add_argument("--model-data", required=True, help="S3 URI to model.tar.gz, e.g., s3://bucket/key/model.tar.gz")
    ap.add_argument("--instance-type", default="ml.m5.large", help="Instance type for endpoint")
    ap.add_argument("--region", default="ca-central-1", help="AWS region")
    args = ap.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)

    model_name = f"{args.endpoint_name}-model"
    config_name = f"{args.endpoint_name}-config"

    # 1) Create or update Model
    print("Creating/Updating Model:", model_name)
    try:
        sm.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": args.image_uri,
                "ModelDataUrl": args.model_data,
                # Optionally, you can pass Environment here for pyfunc server params
            },
            ExecutionRoleArn=args.role_arn,
        )
    except sm.exceptions.ClientError as e:
        if "ConflictException" in str(e) or "already exists" in str(e):
            print("Model exists, updating by deleting and recreating...")
            sm.delete_model(ModelName=model_name)
            sm.create_model(
                ModelName=model_name,
                PrimaryContainer={"Image": args.image_uri, "ModelDataUrl": args.model_data},
                ExecutionRoleArn=args.role_arn,
            )
        else:
            raise

    # 2) Create or update EndpointConfig
    print("Creating/Updating EndpointConfig:", config_name)
    try:
        sm.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InitialInstanceCount": 1,
                    "InstanceType": args.instance_type,
                    "InitialVariantWeight": 1.0,
                }
            ],
        )
    except sm.exceptions.ClientError as e:
        if "ConflictException" in str(e) or "already exists" in str(e):
            print("EndpointConfig exists, updating by deleting and recreating...")
            sm.delete_endpoint_config(EndpointConfigName=config_name)
            sm.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        "VariantName": "AllTraffic",
                        "ModelName": model_name,
                        "InitialInstanceCount": 1,
                        "InstanceType": args.instance_type,
                        "InitialVariantWeight": 1.0,
                    }
                ],
            )
        else:
            raise

    # 3) Create or update Endpoint
    print("Creating/Updating Endpoint:", args.endpoint_name)
    try:
        sm.create_endpoint(
            EndpointName=args.endpoint_name,
            EndpointConfigName=config_name,
        )
    except sm.exceptions.ClientError as e:
        if "already exists" in str(e):
            print("Endpoint exists, updating...")
            sm.update_endpoint(
                EndpointName=args.endpoint_name,
                EndpointConfigName=config_name,
            )
        else:
            raise

    # 4) Wait until endpoint is ready
    status = wait_for_endpoint(sm, args.endpoint_name)
    print("Final endpoint status:", status)
    if status != "InService":
        raise SystemExit("Endpoint failed to reach InService state.")

if __name__ == "__main__":
    main()
