import os
import time
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from sqlalchemy import create_engine, text
from src.config.runtime import get_project_runtime_dbt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HOTPATH_BUILD_SELECT = [
    "dim_date",
    "dim_time",
    "fct_station_status",
    "int_station_status_enriched",
]
DEFAULT_HOTPATH_TEST_SELECT = [
    "fct_station_status",
    "int_station_status_enriched",
]
DEFAULT_FEATURE_BUILD_SELECT = [
    "feat_station_snapshot_5min",
    "feat_station_snapshot_latest",
]
DEFAULT_FEATURE_TEST_SELECT = [
    "feat_station_snapshot_5min",
    "feat_station_snapshot_latest",
]
DEFAULT_WEATHER_REFRESH_SELECT = [
    "dim_weather",
]
DEFAULT_STATION_TOPOLOGY_SELECT = [
    "dim_station",
    "int_station_neighbors",
]
DEFAULT_FEATURE_BUILD_SELECTOR = "hf_feature_build_models"
DEFAULT_HOTPATH_SELECTOR = "hf_station_status_hotpath_models"
DEFAULT_WEATHER_REFRESH_SELECTOR = "weather_refresh_models"
DEFAULT_STATION_TOPOLOGY_SELECTOR = "station_topology_daily_models"
DEFAULT_HOTPATH_TEST_SELECTOR = "hf_station_status_smoke_tests"
DEFAULT_FEATURE_TEST_SELECTOR = "hf_feature_smoke_tests"
DEFAULT_QUALITY_TEST_SELECTOR = "hourly_quality_gate_tests"
DEFAULT_QUALITY_TEST_SELECT = [
    "stg_station_information",
    "stg_station_status",
    "stg_weather_current",
    "stg_weather_hourly",
    "stg_holidays",
    "dim_date",
    "dim_time",
    "dim_weather",
]
DEFAULT_DEEP_QUALITY_TEST_SELECTOR = "daily_deep_quality_tests"
DEFAULT_DEEP_QUALITY_TEST_SELECT = [
    "tag:deep_quality",
    "dim_station",
    "int_station_neighbors",
]
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
        check=False,
        capture_output=True,
        text=True,
        env=env or os.environ.copy(),
    )
    if completed.stdout:
        print(completed.stdout.strip())
    if completed.stderr:
        print(completed.stderr.strip())
    completed.check_returncode()
    return completed


def run_dbt_command(
    action: str,
    project_dir: str,
    profiles_dir: str,
    select_models: Sequence[str] | None = None,
    extra_args: Sequence[str] | None = None,
    selector: str | None = None,
    dbt_vars: dict | None = None,
    indirect_selection: str | None = None,
) -> tuple[subprocess.CompletedProcess, float]:
    command = [
        get_project_runtime_dbt(),
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
    if indirect_selection:
        command.extend(["--indirect-selection", indirect_selection])
    if dbt_vars:
        serialized_vars = {
            key: (
                value.astimezone(timezone.utc).isoformat()
                if isinstance(value, datetime)
                else value
            )
            for key, value in dbt_vars.items()
            if value is not None
        }
        command.extend(["--vars", json.dumps(serialized_vars)])
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
    threads: int = 2,
) -> dict[str, float | str | bool]:
    command = [
        get_project_runtime_dbt(),
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


def run_model_build(
    project_dir: str = "dbt/bikeshare_dbt",
    profiles_dir: str = "dbt",
    select_models: Sequence[str] | None = None,
    selector: str | None = None,
    threads: int = 2,
    dbt_vars: dict | None = None,
) -> dict[str, float | str | bool]:
    build_select = expand_with_parents(select_models) if select_models else None
    _, run_duration = run_dbt_command(
        "run",
        project_dir,
        profiles_dir,
        select_models=build_select,
        selector=selector,
        extra_args=["--fail-fast", "--threads", str(threads)],
        dbt_vars=dbt_vars,
    )
    summary = {
        "completed": True,
        "run_duration_seconds": round(run_duration, 3),
        "threads": threads,
        "selection": selector or " ".join(build_select or []),
    }
    print(f"DBT_MODEL_BUILD_SUMMARY {summary}")
    return summary


def run_feature_model_build(
    project_dir: str = "dbt/bikeshare_dbt",
    profiles_dir: str = "dbt",
    select_models: Sequence[str] | None = None,
    selector: str | None = None,
    threads: int = 2,
    dbt_vars: dict | None = None,
) -> dict[str, float | str | bool]:
    summary = run_model_build(
        project_dir=project_dir,
        profiles_dir=profiles_dir,
        select_models=select_models,
        selector=selector,
        threads=threads,
        dbt_vars=dbt_vars,
    )
    print(f"DBT_FEATURE_BUILD_SUMMARY {summary}")
    return summary


def run_model_tests(
    project_dir: str = "dbt/bikeshare_dbt",
    profiles_dir: str = "dbt",
    select_models: Sequence[str] | None = None,
    selector: str | None = None,
    threads: int = 2,
    dbt_vars: dict | None = None,
    summary_label: str = "DBT_MODEL_TEST_SUMMARY",
    indirect_selection: str | None = None,
) -> dict[str, float | str]:
    _, test_duration = run_dbt_command(
        "test",
        project_dir,
        profiles_dir,
        select_models=select_models,
        selector=selector,
        extra_args=["--fail-fast", "--threads", str(threads)],
        dbt_vars=dbt_vars,
        indirect_selection=indirect_selection,
    )
    summary = {
        "test_duration_seconds": round(test_duration, 3),
        "threads": threads,
        "selection": selector or " ".join(select_models or []),
    }
    print(f"{summary_label} {summary}")
    return summary


def run_feature_smoke_tests(
    project_dir: str = "dbt/bikeshare_dbt",
    profiles_dir: str = "dbt",
    select_models: Sequence[str] | None = None,
    selector: str | None = None,
    threads: int = 2,
    dbt_vars: dict | None = None,
) -> dict[str, float | str]:
    summary = run_model_tests(
        project_dir=project_dir,
        profiles_dir=profiles_dir,
        select_models=select_models,
        selector=selector,
        threads=threads,
        dbt_vars=dbt_vars,
        summary_label="DBT_FEATURE_SMOKE_SUMMARY",
    )
    return summary


def run_quality_tests(
    project_dir: str = "dbt/bikeshare_dbt",
    profiles_dir: str = "dbt",
    select_models: Sequence[str] | None = None,
    selector: str | None = None,
    threads: int = 2,
    dbt_vars: dict | None = None,
) -> dict[str, float | str]:
    summary = run_model_tests(
        project_dir=project_dir,
        profiles_dir=profiles_dir,
        select_models=select_models,
        selector=selector,
        threads=threads,
        dbt_vars=dbt_vars,
        summary_label="DBT_QUALITY_GATE_SUMMARY",
        indirect_selection="cautious",
    )
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


def check_dim_weather_staleness(
    *,
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    schema: str,
    city: str,
    stale_error_minutes: int = 60,
) -> dict[str, object]:
    engine = create_engine(build_postgres_uri(host, port, database, user, password), pool_pre_ping=True)
    now_utc = datetime.now(timezone.utc)
    safe_schema = schema.replace('"', "")
    latest_weather_sql = text(
        f"""
        select
            max(observed_at) as latest_observed_at,
            max(snapshot_bucket_at_utc) as latest_snapshot_bucket_at_utc
        from {safe_schema}.dim_weather
        where city = :city
        """
    )
    with engine.connect() as connection:
        row = connection.execute(latest_weather_sql, {"city": city}).mappings().one()

    latest_observed_at = row["latest_observed_at"]
    latest_snapshot_bucket_at_utc = row["latest_snapshot_bucket_at_utc"]
    if latest_observed_at is None or latest_snapshot_bucket_at_utc is None:
        raise RuntimeError(f"dim_weather staleness check failed: missing curated weather rows for city={city}")

    observed_age_minutes = (now_utc - latest_observed_at).total_seconds() / 60.0
    snapshot_age_minutes = (now_utc - latest_snapshot_bucket_at_utc).total_seconds() / 60.0
    if observed_age_minutes > stale_error_minutes:
        raise RuntimeError(
            "dim_weather staleness check failed: "
            f"city={city} observed_age_minutes={observed_age_minutes:.1f} "
            f"snapshot_age_minutes={snapshot_age_minutes:.1f}"
        )

    summary = {
        "city": city,
        "latest_observed_at_utc": latest_observed_at.isoformat(),
        "latest_snapshot_bucket_at_utc": latest_snapshot_bucket_at_utc.isoformat(),
        "observed_age_minutes": round(observed_age_minutes, 3),
        "snapshot_age_minutes": round(snapshot_age_minutes, 3),
        "stale_error_minutes": stale_error_minutes,
    }
    print(f"DIM_WEATHER_STALENESS_SUMMARY {summary}")
    return summary


def check_dim_station_staleness(
    *,
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    schema: str,
    city: str,
    stale_warn_hours: int = 73,
) -> dict[str, object]:
    engine = create_engine(build_postgres_uri(host, port, database, user, password), pool_pre_ping=True)
    now_utc = datetime.now(timezone.utc)
    safe_schema = schema.replace('"', "")
    latest_station_sql = text(
        f"""
        select max(valid_from_utc) as latest_valid_from_utc
        from {safe_schema}.dim_station
        where city = :city
        """
    )
    with engine.connect() as connection:
        row = connection.execute(latest_station_sql, {"city": city}).mappings().one()

    latest_valid_from_utc = row["latest_valid_from_utc"]
    if latest_valid_from_utc is None:
        raise RuntimeError(f"dim_station staleness check failed: missing station dimension rows for city={city}")

    age_hours = (now_utc - latest_valid_from_utc).total_seconds() / 3600.0
    stale_station_warn = age_hours > stale_warn_hours
    summary = {
        "city": city,
        "latest_valid_from_utc": latest_valid_from_utc.isoformat(),
        "station_age_hours": round(age_hours, 3),
        "stale_warn_hours": stale_warn_hours,
        "stale_station_warn": stale_station_warn,
    }
    print(f"DIM_STATION_STALENESS_SUMMARY {summary}")
    return summary


def run_feature_build(
    project_dir: str = "dbt/bikeshare_dbt",
    profiles_dir: str = "dbt",
    select_models: Sequence[str] | None = None,
    test_models: Sequence[str] | None = None,
    threads: int = 2,
    skip_tests: bool = False,
    dbt_vars: dict | None = None,
) -> dict[str, float | str | bool]:
    model_select = expand_with_parents(select_models or DEFAULT_FEATURE_BUILD_SELECT)
    direct_test_select = list(test_models or DEFAULT_FEATURE_TEST_SELECT)

    _, run_duration = run_dbt_command(
        "run",
        project_dir,
        profiles_dir,
        model_select,
        extra_args=["--fail-fast", "--threads", str(threads)],
        dbt_vars=dbt_vars,
    )

    test_duration = 0.0
    if not skip_tests:
        _, test_duration = run_dbt_command(
            "test",
            project_dir,
            profiles_dir,
            direct_test_select,
            extra_args=["--fail-fast", "--threads", str(threads)],
            dbt_vars=dbt_vars,
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


