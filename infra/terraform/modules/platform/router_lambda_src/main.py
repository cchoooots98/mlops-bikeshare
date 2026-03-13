import json
import os

import boto3


def _resolve_target(event):
    target_name = str(event.get("target_name") or "").strip().lower()
    if target_name in {"bikes", "docks"}:
        return target_name

    raw_predict_bikes = event.get("predict_bikes", True)
    if isinstance(raw_predict_bikes, str):
        raw_predict_bikes = raw_predict_bikes.strip().lower() in {"1", "true", "yes", "y", "on"}
    return "bikes" if raw_predict_bikes else "docks"


def _resolve_environment(event):
    environment = str(event.get("environment") or os.getenv("DEFAULT_ENVIRONMENT", "production")).strip().lower()
    return "prod" if environment == "production" else environment


def _resolve_endpoint(target_name, environment):
    key = f"{target_name}_{environment}".upper()
    endpoint = os.getenv(f"ENDPOINT_{key}")
    if not endpoint:
        raise RuntimeError(f"missing router endpoint mapping for {target_name}/{environment}")
    return endpoint


def handler(event, context):
    target_name = _resolve_target(event or {})
    environment = _resolve_environment(event or {})
    endpoint_name = _resolve_endpoint(target_name, environment)

    payload = {"target_name": target_name, "environment": environment, "request": event or {}}
    runtime = boto3.client("sagemaker-runtime")
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    body = response["Body"].read().decode("utf-8")
    return {
        "ok": True,
        "target_name": target_name,
        "environment": environment,
        "endpoint_name": endpoint_name,
        "body": body,
    }
