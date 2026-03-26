import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.config.naming import DEFAULT_DEPLOYMENT_STATE_ROOT, deployment_state_path, endpoint_name, resolve_target_name
from src.model_target import parse_bool_value

DEFAULT_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parents[1] / "env.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT_RUNTIME_PYTHON = "/opt/project-venv/bin/python"
DEFAULT_PROJECT_RUNTIME_DBT = "/opt/project-venv/bin/dbt"


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


def _env_setting(name: str) -> str | None:
    value = os.getenv(name)
    if value in {None, ""}:
        return None
    return value


def get_project_runtime_python() -> str:
    configured = os.getenv("PROJECT_RUNTIME_PYTHON", "").strip()
    if configured:
        return configured

    if Path(DEFAULT_PROJECT_RUNTIME_PYTHON).exists():
        return DEFAULT_PROJECT_RUNTIME_PYTHON

    return sys.executable


def get_project_runtime_dbt() -> str:
    configured = os.getenv("PROJECT_RUNTIME_DBT", "").strip()
    if configured:
        return configured

    if Path(DEFAULT_PROJECT_RUNTIME_DBT).exists():
        return DEFAULT_PROJECT_RUNTIME_DBT

    discovered = shutil.which("dbt")
    if discovered:
        return discovered

    return "dbt"


def run_project_module(
    module_name: str,
    *,
    args: Sequence[str] | None = None,
    extra_env: dict[str, str] | None = None,
    cwd: str | None = None,
    result_prefix: str | None = None,
) -> object | None:
    env = os.environ.copy()
    env.update(extra_env or {})
    command = [get_project_runtime_python(), "-m", module_name, *(args or [])]
    process = subprocess.Popen(
        command,
        cwd=cwd or str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    stdout_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        stdout_lines.append(line)

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)

    output_lines = [line.strip() for line in stdout_lines if line.strip()]
    if not output_lines:
        return None

    candidate = output_lines[-1]
    if result_prefix:
        prefixed = next((line for line in reversed(output_lines) if line.startswith(result_prefix)), None)
        if prefixed:
            candidate = prefixed[len(result_prefix) :]
        else:
            return None

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return candidate


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

    explicit_target_name = _env_setting("TARGET_NAME")
    explicit_predict_bikes = _env_setting("PREDICT_BIKES")
    configured_target_name = _setting("TARGET_NAME")
    configured_predict_bikes = _setting("PREDICT_BIKES")

    if explicit_target_name is not None:
        target_name = resolve_target_name(target_name=explicit_target_name)
    elif explicit_predict_bikes is not None:
        target_name = resolve_target_name(predict_bikes=explicit_predict_bikes)
    else:
        target_name = resolve_target_name(
            target_name=configured_target_name,
            predict_bikes=configured_predict_bikes if configured_target_name in {None, ""} else None,
        )

    if explicit_predict_bikes is not None:
        predict_bikes = parse_bool_value(explicit_predict_bikes, default=True)
    elif explicit_target_name is not None or configured_target_name not in {None, ""}:
        predict_bikes = target_name == "bikes"
    else:
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
        sm_endpoint=str(
            _setting("SM_ENDPOINT", endpoint_name(target_name=target_name, environment=serving_environment))
        ),
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
