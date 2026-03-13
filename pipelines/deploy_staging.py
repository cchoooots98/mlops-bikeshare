import argparse
from pathlib import Path
from typing import Sequence

from pipelines.deploy_via_sagemaker_sdk import main as deploy_main
from src.config import deployment_state_path


def _infer_target_name(endpoint_name: str) -> str:
    normalized = endpoint_name.lower()
    if "-docks-" in normalized:
        return "docks"
    return "bikes"


def parse_args(argv: Sequence[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(description="Deploy a packaged candidate model to the staging environment.")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--package-s3-uri", required=True)
    parser.add_argument("--package-dir", default=None)
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--region", default="eu-west-3")
    parser.add_argument("--target-name", default=None, choices=["bikes", "docks"])
    parser.add_argument("--deployment-state-root", default="model_dir/deployments")
    parser.add_argument("--deployment-state-path", default=None)
    args = parser.parse_args(argv)
    target_name = args.target_name or _infer_target_name(args.endpoint_name)
    deployment_state = args.deployment_state_path or str(
        deployment_state_path(
            target_name=target_name,
            environment="staging",
            root=Path(args.deployment_state_root),
        )
    )
    forwarded = [
        "--endpoint-name",
        args.endpoint_name,
        "--role-arn",
        args.role_arn,
        "--image-uri",
        args.image_uri,
        "--package-s3-uri",
        args.package_s3_uri,
        "--instance-type",
        args.instance_type,
        "--region",
        args.region,
        "--environment",
        "staging",
        "--deployment-state-path",
        deployment_state,
    ]
    if args.package_dir:
        forwarded.extend(["--package-dir", args.package_dir])
    return forwarded


def main(argv: Sequence[str] | None = None) -> dict:
    return deploy_main(parse_args(argv))


if __name__ == "__main__":
    main()
