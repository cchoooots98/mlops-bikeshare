from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.config import resolve_predict_bikes, resolve_target_name


@dataclass(frozen=True)
class RouterRequest:
    predict_bikes: bool
    target_name: str
    environment: str
    sm_endpoint_override: str | None = None
    validate_only: bool = False


def parse_router_request(event: Mapping[str, Any] | None) -> RouterRequest:
    payload = event if isinstance(event, Mapping) else {}
    target_name = resolve_target_name(
        predict_bikes=payload.get("predict_bikes"),
        target_name=payload.get("target_name"),
    )
    predict_bikes = resolve_predict_bikes(
        predict_bikes=payload.get("predict_bikes"),
        target_name=payload.get("target_name"),
    )
    endpoint_override = payload.get("sm_endpoint")
    return RouterRequest(
        predict_bikes=predict_bikes,
        target_name=target_name,
        environment=str(payload.get("environment") or "production"),
        sm_endpoint_override=str(endpoint_override) if endpoint_override not in {None, ""} else None,
        validate_only=bool(payload.get("validate_only", False)),
    )
