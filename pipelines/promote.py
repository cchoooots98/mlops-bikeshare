# pipelines/promote.py
# Purpose: Promote a model version or the current Staging to Production
# in MLflow Registry. Optionally deploy a separate "prod" endpoint to
# support blue/green style handover.
#
# Usage (PowerShell):
#   # Promote the latest Staging to Production:
#   python pipelines/promote.py `
#       --model-name bikeshare_risk `
#       --from-stage Staging `
#       --to-stage Production
#
#   # Promote a specific version:
#   python pipelines/promote.py `
#       --model-name bikeshare_risk `
#       --version 5 `
#       --to-stage Production
#
#   # Optional: also (re)deploy Production endpoint after promotion:
#   #   (uncomment code in the script where indicated to call mlflow.sagemaker.deploy)

import argparse

import mlflow


def main():
    parser = argparse.ArgumentParser(description="Promote MLflow model version or stage.")
    parser.add_argument("--model-name", required=True, help="Registry name, e.g., 'bikeshare_risk'.")
    parser.add_argument(
        "--version", default=None, help="Model version to promote (mutually exclusive with --from-stage)."
    )
    parser.add_argument(
        "--from-stage", default=None, help="Stage to promote from (e.g., Staging), if --version not given."
    )
    parser.add_argument("--to-stage", required=True, help="Target stage, e.g., Production.")
    args = parser.parse_args()

    client = mlflow.tracking.MlflowClient()

    # Resolve version: either explicit --version or "latest from-stage"
    if args.version and args.from_stage:
        raise ValueError("Use either --version or --from-stage, not both.")
    if args.version:
        version = args.version
    else:
        # Get latest version currently in the given from-stage
        versions = client.get_latest_versions(args.model_name, stages=[args.from_stage])
        if not versions:
            raise RuntimeError(f"No versions found in stage {args.from_stage}.")
        version = versions[0].version

    # Transition stage (no auto-archive for a gentler workflow)
    client.transition_model_version_stage(
        name=args.model_name,
        version=version,
        stage=args.to_stage,
        archive_existing_versions=False,
    )
    print(f"Promoted: name={args.model_name}, version={version}, to={args.to_stage}")

    # OPTIONAL: deploy a prod endpoint here (blue/green style)
    # Example (uncomment to use):
    # mlflow.sagemaker.deploy(
    #   app_name="bikeshare-prod",
    #   model_uri=f"models:/{args.model_name}/{args.to_stage}",
    #   mode="create",
    #   region_name="ca-central-1",
    #   execution_role_arn="arn:aws:iam::387706002632:role/mlops-bikeshare-sagemaker-exec",
    #   instance_type="ml.m5.large",
    #   timeout_seconds=600,
    # )


if __name__ == "__main__":
    main()
