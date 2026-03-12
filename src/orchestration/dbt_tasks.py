import os
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from sqlalchemy import create_engine, text

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_BUILD_SELECT = [
    "feat_station_snapshot_5min",
    "feat_station_snapshot_latest",
]
DEFAULT_FEATURE_BUILD_SELECTOR = "hf_feature_build_models"
DEFAULT_FEATURE_TEST_SELECTOR = "hf_smoke_tests"
DEFAULT_QUALITY_MODEL_SELECTOR = "quality_models"
DEFAULT_QUALITY_TEST_SELECTOR = "quality_gate_tests"
DEFAULT_SOURCE_FRESHNESS_SELECT = [
    "source:raw_staging.stg_station_status",
    "source:raw_staging.stg_station_information",
    "source:raw_staging.stg_weather_current",
    "source:raw_staging.stg_weather_hourly",
    "source:raw_staging.stg_holidays",
]


def parse_select_models(raw_value: str | None, default_models: Sequence[str] | None = None) -> list[str]:
    if raw_value and raw_value.strip():
        return [item for item in raw_value.split() if item]
    return list(default_models or DEFAULT_FEATURE_BUILD_SELECT)


def parse_bool(raw_value: str | None, default: bool = False) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_selector(raw_value: str | None, default_selector: str) -> str:
    if raw_value and raw_value.strip():
        return raw_value.strip()
    return default_selector


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
    select_models: Sequence[str] | None = None,
    extra_args: Sequence[str] | None = None,
    selector: str | None = None,
) -> tuple[subprocess.CompletedProcess, float]:
    command = [
        "dbt",
        action,
        "--project-dir",
        project_dir,
        "--profiles-dir",
        profiles_dir,
    ]
    if selector:
        command.extend(["--selector", selector])
    elif select_models:
        command.extend(["--select", *select_models])
    if extra_args:
        command.extend(extra_args)
    started = time.perf_counter()
    completed = run_command(command)
    duration_seconds = time.perf_counter() - started
    selection = selector or " ".join(select_models or [])
    print(
        "DBT_COMMAND_TIMING "
        f"action={action} "
        f"duration_seconds={duration_seconds:.3f} "
        f"selection={selection}"
    )
    return completed, duration_seconds


def run_dbt_source_freshness(
    project_dir: str = "dbt/bikeshare_dbt",
    profiles_dir: str = "dbt",
    select_models: Sequence[str] | None = None,
    threads: int = 1,
) -> dict[str, float | str | bool]:
    command = [
        "dbt",
        "source",
        "freshness",
        "--project-dir",
        project_dir,
        "--profiles-dir",
        profiles_dir,
        "--select",
        *(select_models or DEFAULT_SOURCE_FRESHNESS_SELECT),
        "--threads",
        str(threads),
    ]
    started = time.perf_counter()
    completed = run_command(command)
    duration_seconds = time.perf_counter() - started
    summary = {
        "duration_seconds": round(duration_seconds, 3),
        "selection": " ".join(select_models or DEFAULT_SOURCE_FRESHNESS_SELECT),
    }
    print(f"DBT_SOURCE_FRESHNESS_SUMMARY {summary}")
    return {"completed": completed.returncode == 0, **summary}


def build_postgres_uri(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
) -> str:
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


def run_feature_model_build(
    project_dir: str = "dbt/bikeshare_dbt",
    profiles_dir: str = "dbt",
    selector: str = DEFAULT_FEATURE_BUILD_SELECTOR,
    threads: int = 1,
) -> dict[str, float | str]:
    _, run_duration = run_dbt_command(
        "run",
        project_dir,
        profiles_dir,
        selector=selector,
        extra_args=["--fail-fast", "--threads", str(threads)],
    )
    summary = {
        "run_duration_seconds": round(run_duration, 3),
        "threads": threads,
        "selector": selector,
    }
    print(f"DBT_FEATURE_BUILD_SUMMARY {summary}")
    return summary


def run_feature_smoke_tests(
    project_dir: str = "dbt/bikeshare_dbt",
    profiles_dir: str = "dbt",
    selector: str = DEFAULT_FEATURE_TEST_SELECTOR,
    threads: int = 1,
) -> dict[str, float | str]:
    _, test_duration = run_dbt_command(
        "test",
        project_dir,
        profiles_dir,
        selector=selector,
        extra_args=["--fail-fast", "--threads", str(threads)],
    )
    summary = {
        "test_duration_seconds": round(test_duration, 3),
        "threads": threads,
        "selector": selector,
    }
    print(f"DBT_FEATURE_SMOKE_SUMMARY {summary}")
    return summary


def run_quality_models(
    project_dir: str = "dbt/bikeshare_dbt",
    profiles_dir: str = "dbt",
    selector: str = DEFAULT_QUALITY_MODEL_SELECTOR,
    threads: int = 1,
) -> dict[str, float | str]:
    _, duration = run_dbt_command(
        "run",
        project_dir,
        profiles_dir,
        selector=selector,
        extra_args=["--fail-fast", "--threads", str(threads)],
    )
    summary = {
        "run_duration_seconds": round(duration, 3),
        "threads": threads,
        "selector": selector,
    }
    print(f"DBT_QUALITY_MODEL_SUMMARY {summary}")
    return summary


def run_quality_tests(
    project_dir: str = "dbt/bikeshare_dbt",
    profiles_dir: str = "dbt",
    selector: str = DEFAULT_QUALITY_TEST_SELECTOR,
    threads: int = 1,
) -> dict[str, float | str]:
    _, duration = run_dbt_command(
        "test",
        project_dir,
        profiles_dir,
        selector=selector,
        extra_args=["--fail-fast", "--threads", str(threads)],
    )
    summary = {
        "test_duration_seconds": round(duration, 3),
        "threads": threads,
        "selector": selector,
    }
    print(f"DBT_QUALITY_GATE_SUMMARY {summary}")
    return summary


def check_weather_semantic_freshness(
    *,
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    city: str,
    current_warn_minutes: int = 30,
    current_error_minutes: int = 120,
    hourly_warn_minutes: int = 30,
    hourly_error_minutes: int = 120,
    min_forecast_coverage_minutes: int = 30,
) -> dict[str, object]:
    engine = create_engine(build_postgres_uri(host, port, database, user, password), pool_pre_ping=True)
    now_utc = datetime.now(timezone.utc)

    current_sql = text(
        """
        select max(observed_at) as current_observed_at
        from public.stg_weather_current
        where city = :city
        """
    )
    hourly_sql = text(
        """
        select
            max(snapshot_bucket_at) as hourly_snapshot_at,
            max(forecast_at) as hourly_forecast_at
        from public.stg_weather_hourly
        where city = :city
        """
    )
    with engine.connect() as connection:
        current_row = connection.execute(current_sql, {"city": city}).mappings().one()
        hourly_row = connection.execute(hourly_sql, {"city": city}).mappings().one()

    current_observed_at = current_row["current_observed_at"]
    hourly_snapshot_at = hourly_row["hourly_snapshot_at"]
    hourly_forecast_at = hourly_row["hourly_forecast_at"]
    if current_observed_at is None or hourly_snapshot_at is None or hourly_forecast_at is None:
        raise RuntimeError(f"weather semantic freshness check failed: missing weather records for city={city}")

    current_age_minutes = (now_utc - current_observed_at).total_seconds() / 60.0
    hourly_age_minutes = (now_utc - hourly_snapshot_at).total_seconds() / 60.0
    forecast_coverage_minutes = (hourly_forecast_at - now_utc).total_seconds() / 60.0

    stale_weather_warn = False
    if (
        current_age_minutes > current_error_minutes
        or hourly_age_minutes > hourly_error_minutes
        or forecast_coverage_minutes < 0
    ):
        raise RuntimeError(
            "weather semantic freshness check failed: "
            f"city={city} current_age_minutes={current_age_minutes:.1f} "
            f"hourly_age_minutes={hourly_age_minutes:.1f} "
            f"forecast_coverage_minutes={forecast_coverage_minutes:.1f}"
        )

    if (
        current_age_minutes > current_warn_minutes
        or hourly_age_minutes > hourly_warn_minutes
        or forecast_coverage_minutes < min_forecast_coverage_minutes
    ):
        stale_weather_warn = True

    summary = {
        "city": city,
        "current_observed_at_utc": current_observed_at.isoformat(),
        "hourly_snapshot_at_utc": hourly_snapshot_at.isoformat(),
        "hourly_forecast_at_utc": hourly_forecast_at.isoformat(),
        "current_age_minutes": round(current_age_minutes, 3),
        "hourly_age_minutes": round(hourly_age_minutes, 3),
        "forecast_coverage_minutes": round(forecast_coverage_minutes, 3),
        "stale_weather_warn": stale_weather_warn,
    }
    print(f"WEATHER_FRESHNESS_SUMMARY {summary}")
    return summary


def run_feature_build(
    project_dir: str = "dbt/bikeshare_dbt",
    profiles_dir: str = "dbt",
    select_models: Sequence[str] | None = None,
    test_models: Sequence[str] | None = None,
    threads: int = 1,
    skip_tests: bool = False,
) -> dict[str, float | str | bool]:
    model_select = expand_with_parents(select_models or DEFAULT_FEATURE_BUILD_SELECT)
    direct_test_select = list(test_models or DEFAULT_FEATURE_BUILD_SELECT)

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
