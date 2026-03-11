import os
import subprocess
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_BUILD_SELECT = [
    "dim_station",
    "dim_time",
    "dim_weather",
    "dim_date",
    "fct_station_status",
    "int_station_neighbors",
    "int_station_status_enriched",
    "feat_station_snapshot_5min",
    "feat_station_snapshot_latest",
]


def parse_select_models(raw_value: str | None, default_models: Sequence[str] | None = None) -> list[str]:
    if raw_value and raw_value.strip():
        return [item for item in raw_value.split() if item]
    return list(default_models or DEFAULT_FEATURE_BUILD_SELECT)


def run_command(command: Sequence[str], env: dict | None = None) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
        env=env or os.environ.copy(),
    )
    if completed.stdout:
        print(completed.stdout.strip())
    if completed.stderr:
        print(completed.stderr.strip())
    return completed


def run_dbt_command(
    action: str,
    project_dir: str,
    profiles_dir: str,
    select_models: Sequence[str],
    extra_args: Sequence[str] | None = None,
) -> subprocess.CompletedProcess:
    command = [
        "dbt",
        action,
        "--project-dir",
        project_dir,
        "--profiles-dir",
        profiles_dir,
        "--select",
        *select_models,
    ]
    if extra_args:
        command.extend(extra_args)
    return run_command(command)


def run_feature_build(
    project_dir: str = "dbt/bikeshare_dbt",
    profiles_dir: str = "dbt",
    select_models: Sequence[str] | None = None,
) -> None:
    models = list(select_models or DEFAULT_FEATURE_BUILD_SELECT)
    run_dbt_command("run", project_dir, profiles_dir, models, extra_args=["--fail-fast"])
    run_dbt_command("test", project_dir, profiles_dir, models, extra_args=["--fail-fast"])
