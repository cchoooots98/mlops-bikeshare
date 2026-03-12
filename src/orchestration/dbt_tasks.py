import os
import time
import subprocess
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_BUILD_SELECT = [
    "fct_station_inventory_capacity_anomalies",
    "feat_station_snapshot_5min",
    "feat_station_snapshot_latest",
]
DEFAULT_FEATURE_TEST_SELECT = list(DEFAULT_FEATURE_BUILD_SELECT)


def parse_select_models(raw_value: str | None, default_models: Sequence[str] | None = None) -> list[str]:
    if raw_value and raw_value.strip():
        return [item for item in raw_value.split() if item]
    return list(default_models or DEFAULT_FEATURE_BUILD_SELECT)


def parse_bool(raw_value: str | None, default: bool = False) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def expand_with_parents(select_models: Sequence[str]) -> list[str]:
    expanded = []
    for item in select_models:
        if item.startswith(("+", "@", "path:", "tag:", "source:", "fqn:")):
            expanded.append(item)
        else:
            expanded.append(f"+{item}")
    return expanded


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
) -> tuple[subprocess.CompletedProcess, float]:
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
    started = time.perf_counter()
    completed = run_command(command)
    duration_seconds = time.perf_counter() - started
    print(
        "DBT_COMMAND_TIMING "
        f"action={action} "
        f"duration_seconds={duration_seconds:.3f} "
        f"select={' '.join(select_models)}"
    )
    return completed, duration_seconds


def run_feature_build(
    project_dir: str = "dbt/bikeshare_dbt",
    profiles_dir: str = "dbt",
    select_models: Sequence[str] | None = None,
    test_models: Sequence[str] | None = None,
    threads: int = 1,
    skip_tests: bool = False,
) -> dict[str, float | str | bool]:
    model_select = expand_with_parents(select_models or DEFAULT_FEATURE_BUILD_SELECT)
    direct_test_select = list(test_models or DEFAULT_FEATURE_TEST_SELECT)

    _, run_duration = run_dbt_command(
        "run",
        project_dir,
        profiles_dir,
        model_select,
        extra_args=["--fail-fast", "--threads", str(threads)],
    )

    test_duration = 0.0
    if not skip_tests:
        _, test_duration = run_dbt_command(
            "test",
            project_dir,
            profiles_dir,
            direct_test_select,
            extra_args=["--fail-fast", "--threads", str(threads)],
        )

    total_duration = run_duration + test_duration
    summary = {
        "run_duration_seconds": round(run_duration, 3),
        "test_duration_seconds": round(test_duration, 3),
        "total_duration_seconds": round(total_duration, 3),
        "threads": threads,
        "tests_skipped": skip_tests,
        "run_select": " ".join(model_select),
        "test_select": " ".join(direct_test_select),
    }
    print(f"DBT_FEATURE_BUILD_SUMMARY {summary}")
    return summary
