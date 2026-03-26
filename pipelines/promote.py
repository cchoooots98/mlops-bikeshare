import argparse
import json
from datetime import datetime, timezone
from typing import Sequence

import mlflow
from src.model_package import load_deployment_state, write_deployment_state


def resolve_version_from_alias(client: mlflow.tracking.MlflowClient, model_name: str, alias: str) -> str:
    if hasattr(client, "get_model_version_by_alias"):
        return str(client.get_model_version_by_alias(model_name, alias).version)
    for version in client.search_model_versions(f"name = '{model_name}'"):
        alias_tag = version.tags.get(f"alias.{alias}") if getattr(version, "tags", None) else None
        if alias_tag == "true":
            return str(version.version)
    raise RuntimeError(f"No version found for alias {alias}.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote a deployed package reference and optionally transition MLflow stage."
    )
    parser.add_argument("--source-deployment-state-path", required=True)
    parser.add_argument("--target-deployment-state-path", required=True)
    parser.add_argument("--target-environment", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--alias", default=None)
    parser.add_argument("--to-stage", default=None)
    return parser.parse_args(argv)


def maybe_transition_registry(
    model_name: str | None, version: str | None, alias: str | None, to_stage: str | None
) -> str | None:
    if not to_stage:
        return version
    if not model_name:
        raise ValueError("--model-name is required when --to-stage is used")
    client = mlflow.tracking.MlflowClient()
    resolved_version = version or (resolve_version_from_alias(client, model_name, alias) if alias else None)
    if not resolved_version:
        raise ValueError("provide --version or --alias when promoting an MLflow stage")
    client.transition_model_version_stage(
        name=model_name,
        version=resolved_version,
        stage=to_stage,
        archive_existing_versions=False,
    )
    return resolved_version


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    resolved_version = maybe_transition_registry(args.model_name, args.version, args.alias, args.to_stage)
    state = load_deployment_state(args.source_deployment_state_path)
    promoted_state = {
        **state,
        "environment": args.target_environment,
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "promote",
    }
    if args.model_name:
        promoted_state["registered_model_name"] = args.model_name
    if resolved_version:
        promoted_state["registered_version"] = resolved_version
    output_path = write_deployment_state(args.target_deployment_state_path, promoted_state)
    result = {
        "source_deployment_state_path": args.source_deployment_state_path,
        "target_deployment_state_path": output_path,
        "target_environment": args.target_environment,
        "registered_model_name": promoted_state.get("registered_model_name"),
        "registered_version": promoted_state.get("registered_version"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
