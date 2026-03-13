import argparse
import json
from typing import Sequence

import mlflow

REGISTER_RESULT_PREFIX = "REGISTER_RESULT_JSON::"


def parse_version_tags(raw_tags: Sequence[str] | None) -> dict[str, str]:
    tags = {}
    for item in raw_tags or []:
        if "=" not in item:
            raise ValueError(f"invalid --version-tag value: {item}")
        key, value = item.split("=", 1)
        tags[key] = value
    return tags


def set_alias_or_fallback(client: mlflow.tracking.MlflowClient, model_name: str, version: str, alias: str) -> None:
    if hasattr(client, "set_registered_model_alias"):
        client.set_registered_model_alias(model_name, alias, version)
        return
    client.set_model_version_tag(model_name, version, f"alias.{alias}", "true")


def register_model_version(
    run_id: str,
    model_name: str,
    artifact_path: str = "model",
    stage: str | None = None,
    alias: str | None = None,
    version_tags: dict[str, str] | None = None,
) -> dict:
    source = f"runs:/{run_id}/{artifact_path}"
    model_version = mlflow.register_model(model_uri=source, name=model_name)
    client = mlflow.tracking.MlflowClient()

    if stage:
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage=stage,
            archive_existing_versions=False,
        )

    if alias:
        set_alias_or_fallback(client, model_name, str(model_version.version), alias)

    for key, value in (version_tags or {}).items():
        client.set_model_version_tag(model_name, model_version.version, key, value)

    result = {
        "run_id": run_id,
        "model_name": model_name,
        "artifact_path": artifact_path,
        "version": str(model_version.version),
        "stage": stage,
        "alias": alias,
        "version_tags": version_tags or {},
    }
    print(REGISTER_RESULT_PREFIX + json.dumps(result, sort_keys=True))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register an MLflow model version from a run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--artifact-path", default="model")
    parser.add_argument("--stage", default=None)
    parser.add_argument("--alias", default=None)
    parser.add_argument("--version-tag", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    return register_model_version(
        run_id=args.run_id,
        model_name=args.model_name,
        artifact_path=args.artifact_path,
        stage=args.stage,
        alias=args.alias,
        version_tags=parse_version_tags(args.version_tag),
    )


if __name__ == "__main__":
    main()
