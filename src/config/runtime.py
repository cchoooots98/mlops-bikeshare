import json
import os
from dataclasses import dataclass
from pathlib import Path

from src.config.naming import DEFAULT_DEPLOYMENT_STATE_ROOT, deployment_state_path, endpoint_name, resolve_target_name
from src.model_target import parse_bool_value


DEFAULT_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parents[1] / "env.json"


@dataclass(frozen=True)
class RuntimeSettings:
    aws_region: str
    city: str
    bucket: str
    sm_endpoint: str
    serving_environment: str
    target_name: str
    pg_host: str
    pg_port: int
    pg_db: str
    pg_user: str
    pg_password: str
    pg_schema: str
    training_feature_table: str
    online_feature_table: str
    model_package_dir: str | None
    deployment_state_root: str
    deployment_state_path: str
    predict_bikes: bool = True
    cw_namespace: str = "Bikeshare/Model"
    dev_mode: bool = False


def _load_local_defaults() -> dict:
    if not DEFAULT_RUNTIME_CONFIG_PATH.exists():
        return {}
    with DEFAULT_RUNTIME_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return raw.get("Variables", raw)


def _setting(name: str, default: str | None = None) -> str | None:
    for key in (name,):
        value = os.getenv(key)
        if value not in {None, ""}:
            return value

    defaults = _load_local_defaults()
    for key in (name,):
        value = defaults.get(key)
        if value not in {None, ""}:
            return value
        lower = key.lower()
        value = defaults.get(lower)
        if value not in {None, ""}:
            return value
    return default


def load_runtime_settings() -> RuntimeSettings:
    pg_host = _setting("PGHOST")
    pg_db = _setting("PGDATABASE")
    pg_user = _setting("PGUSER")
    pg_password = _setting("PGPASSWORD")
    missing = [
        name
        for name, value in (
            ("PGHOST", pg_host),
            ("PGDATABASE", pg_db),
            ("PGUSER", pg_user),
            ("PGPASSWORD", pg_password),
        )
        if value in {None, ""}
    ]
    if missing:
        raise ValueError(f"missing required runtime settings: {missing}")

    target_name = resolve_target_name(
        predict_bikes=_setting("PREDICT_BIKES"),
        target_name=_setting("TARGET_NAME"),
    )
    predict_bikes = parse_bool_value(_setting("PREDICT_BIKES", str(target_name == "bikes").lower()), default=True)
    serving_environment = str(_setting("SERVING_ENVIRONMENT", "local"))
    deployment_root = str(_setting("DEPLOYMENT_STATE_ROOT", str(DEFAULT_DEPLOYMENT_STATE_ROOT)))
    resolved_deployment_state_path = _setting("DEPLOYMENT_STATE_PATH")
    if resolved_deployment_state_path in {None, ""}:
        resolved_deployment_state_path = str(
            deployment_state_path(
                target_name=target_name,
                environment=serving_environment,
                root=deployment_root,
            )
        )

    return RuntimeSettings(
        aws_region=str(_setting("AWS_REGION", "eu-west-3")),
        city=str(_setting("CITY", "paris")),
        bucket=str(_setting("BUCKET", "")),
        sm_endpoint=str(_setting("SM_ENDPOINT", endpoint_name(target_name=target_name, environment=serving_environment))),
        serving_environment=serving_environment,
        target_name=target_name,
        pg_host=str(pg_host),
        pg_port=int(_setting("PGPORT", "5432")),
        pg_db=str(pg_db),
        pg_user=str(pg_user),
        pg_password=str(pg_password),
        pg_schema=str(_setting("PGSCHEMA", "analytics")),
        training_feature_table=str(_setting("FEATURE_TABLE", "feat_station_snapshot_5min")),
        online_feature_table=str(_setting("ONLINE_FEATURE_TABLE", "feat_station_snapshot_latest")),
        model_package_dir=_setting("MODEL_PACKAGE_DIR"),
        deployment_state_root=deployment_root,
        deployment_state_path=str(_setting("DEPLOYMENT_STATE_PATH", resolved_deployment_state_path)),
        predict_bikes=predict_bikes,
        cw_namespace=str(_setting("CW_NS", "Bikeshare/Model")),
        dev_mode=parse_bool_value(_setting("DEV_MODE", "false"), default=False),
    )
