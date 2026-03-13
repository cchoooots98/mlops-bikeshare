import json
import os
from dataclasses import dataclass
from pathlib import Path

from src.model_package import DEFAULT_DEPLOYMENT_STATE_PATH
from src.model_target import parse_bool_value


DEFAULT_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parents[1] / "env.json"


@dataclass(frozen=True)
class RuntimeSettings:
    aws_region: str
    city: str
    bucket: str
    sm_endpoint: str
    pg_host: str
    pg_port: int
    pg_db: str
    pg_user: str
    pg_password: str
    pg_schema: str
    training_feature_table: str
    online_feature_table: str
    model_package_dir: str | None
    deployment_state_path: str
    predict_bikes: bool = True
    cw_namespace: str = "Bikeshare/Model"
    athena_output: str | None = None
    athena_workgroup: str | None = None
    athena_database: str | None = None


def _load_local_defaults() -> dict:
    if not DEFAULT_RUNTIME_CONFIG_PATH.exists():
        return {}
    with DEFAULT_RUNTIME_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return raw.get("Variables", raw)


def _setting(name: str, default: str | None = None, *, aliases: tuple[str, ...] = ()) -> str | None:
    for key in (name, *aliases):
        value = os.getenv(key)
        if value not in {None, ""}:
            return value

    defaults = _load_local_defaults()
    for key in (name, *aliases):
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

    return RuntimeSettings(
        aws_region=str(_setting("AWS_REGION", "eu-west-3", aliases=("REGION",))),
        city=str(_setting("CITY", "paris")),
        bucket=str(_setting("BUCKET", "")),
        sm_endpoint=str(_setting("SM_ENDPOINT", "bikeshare-prod")),
        pg_host=str(pg_host),
        pg_port=int(_setting("PGPORT", "5432")),
        pg_db=str(pg_db),
        pg_user=str(pg_user),
        pg_password=str(pg_password),
        pg_schema=str(_setting("PGSCHEMA", "analytics", aliases=("TRAIN_PG_SCHEMA",))),
        training_feature_table=str(
            _setting("FEATURE_TABLE", "feat_station_snapshot_5min", aliases=("TRAIN_FEATURE_TABLE",))
        ),
        online_feature_table=str(
            _setting("ONLINE_FEATURE_TABLE", "feat_station_snapshot_latest", aliases=("SERVING_FEATURE_TABLE",))
        ),
        model_package_dir=_setting("MODEL_PACKAGE_DIR"),
        deployment_state_path=str(
            _setting(
                "DEPLOYMENT_STATE_PATH",
                str(DEFAULT_DEPLOYMENT_STATE_PATH),
                aliases=("MODEL_METADATA_PATH", "RETRAIN_MANIFEST_PATH"),
            )
        ),
        predict_bikes=parse_bool_value(_setting("PREDICT_BIKES", "true"), default=True),
        cw_namespace=str(_setting("CW_NS", "Bikeshare/Model")),
        athena_output=_setting("ATHENA_OUTPUT"),
        athena_workgroup=_setting("ATHENA_WORKGROUP"),
        athena_database=_setting("ATHENA_DATABASE"),
    )
