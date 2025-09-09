# pipelines/register_model.py
# Purpose: Register a trained model (artifact path "model") from an MLflow run
# into the MLflow Model Registry, and (optionally) transition it to a stage.
#
# Usage (PowerShell):
#   python pipelines/register_model.py `
#       --run-id <RUN_ID_FROM_TRAINING> `
#       --model-name bikeshare_risk `
#       --stage Staging
#
# Notes:
# - This script assumes $env:MLFLOW_TRACKING_URI is set (e.g., sqlite:///mlflow.db).
# - "model-name" becomes the registry name (shared across all versions).

import argparse

import mlflow


def main():
    parser = argparse.ArgumentParser(description="Register an MLflow model version from a run.")
    parser.add_argument("--run-id", required=True, help="MLflow run_id that produced the 'model' artifact.")
    parser.add_argument("--model-name", required=True, help="MLflow Model Registry name, e.g., 'bikeshare_risk'.")
    parser.add_argument(
        "--stage", default=None, help="Optional stage to transition to, e.g., 'Staging' or 'Production'."
    )
    args = parser.parse_args()

    # Build the runs:/ URI pointing to the logged model artifact.
    source = f"runs:/{args.run_id}/model"

    # Create a new version in the Model Registry (or use existing name, version increments automatically).
    mv = mlflow.register_model(model_uri=source, name=args.model_name)
    print(f"Registered model version: name={args.model_name}, version={mv.version}")

    # Optionally, set the stage (e.g., mark this version as Staging)
    if args.stage:
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=args.model_name,
            version=mv.version,
            stage=args.stage,
            archive_existing_versions=False,  # do not auto-archive older versions
        )
        print(f"Transitioned model to stage: {args.stage}")


if __name__ == "__main__":
    main()
