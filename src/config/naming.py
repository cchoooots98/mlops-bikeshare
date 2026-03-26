from __future__ import annotations

from pathlib import Path

from src.model_target import parse_bool_value, target_spec_from_name, target_spec_from_predict_bikes

DEFAULT_PROJECT_SLUG = "bikeshare"
DEFAULT_DEPLOYMENT_STATE_ROOT = Path("model_dir") / "deployments"


def resolve_target_name(*, predict_bikes: bool | str | None = None, target_name: str | None = None) -> str:
    if predict_bikes is not None:
        return target_spec_from_predict_bikes(parse_bool_value(predict_bikes)).target_name
    if target_name:
        return target_spec_from_name(target_name).target_name
    return target_spec_from_predict_bikes(True).target_name


def resolve_predict_bikes(*, predict_bikes: bool | str | None = None, target_name: str | None = None) -> bool:
    return target_spec_from_name(
        resolve_target_name(predict_bikes=predict_bikes, target_name=target_name)
    ).predict_bikes


def prediction_key(city: str, dt: str, target_name: str) -> str:
    return f"inference/target={resolve_target_name(target_name=target_name)}/city={city}/dt={dt}/predictions.parquet"


def prediction_prefix(city: str, target_name: str) -> str:
    return f"inference/target={resolve_target_name(target_name=target_name)}/city={city}/dt="


def quality_key(city: str, dt: str, target_name: str) -> str:
    ds = dt[:10]
    return f"monitoring/quality/target={resolve_target_name(target_name=target_name)}/city={city}/ds={ds}/part-{dt}.parquet"


def quality_prefix(city: str, target_name: str) -> str:
    return f"monitoring/quality/target={resolve_target_name(target_name=target_name)}/city={city}"


def deployment_state_path(
    *,
    target_name: str,
    environment: str,
    root: str | Path | None = None,
) -> Path:
    base_dir = Path(root) if root is not None else DEFAULT_DEPLOYMENT_STATE_ROOT
    return base_dir / resolve_target_name(target_name=target_name) / f"{environment}.json"


def endpoint_name(
    *,
    target_name: str,
    environment: str,
    project_slug: str = DEFAULT_PROJECT_SLUG,
) -> str:
    normalized_environment = environment.strip().lower()
    if normalized_environment == "production":
        normalized_environment = "prod"
    return f"{project_slug}-{resolve_target_name(target_name=target_name)}-{normalized_environment}"
