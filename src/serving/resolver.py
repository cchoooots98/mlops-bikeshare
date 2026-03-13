from __future__ import annotations

from pathlib import Path
from typing import Mapping

from src.config import deployment_state_path, endpoint_name, resolve_target_name
from src.model_target import target_spec_from_name


def resolve_target(event: Mapping[str, object] | None = None, *, predict_bikes: object | None = None, target_name: str | None = None):
    payload = event if isinstance(event, Mapping) else {}
    resolved_target_name = resolve_target_name(
        predict_bikes=payload.get("predict_bikes", predict_bikes),
        target_name=str(payload.get("target_name", target_name) or "") or None,
    )
    return target_spec_from_name(resolved_target_name)


def resolve_deployment_state_path(
    *,
    target_name: str,
    environment: str,
    deployment_state_root: str | Path,
) -> Path:
    return deployment_state_path(
        target_name=target_name,
        environment=environment,
        root=deployment_state_root,
    )


def resolve_endpoint_name(
    *,
    target_name: str,
    environment: str,
    project_slug: str = "bikeshare",
) -> str:
    return endpoint_name(target_name=target_name, environment=environment, project_slug=project_slug)
