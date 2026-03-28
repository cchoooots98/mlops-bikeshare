import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.features.postgres_store import PostgresFeatureConfig, create_pg_engine, validate_identifier
from src.features.schema import FEATURE_COLUMNS
from src.monitoring.metrics.metrics_helper import put_metrics_bulk

CORE_PSI_FEATURE_COLUMNS = [
    "util_bikes",
    "util_docks",
    "delta_bikes_5m",
    "delta_docks_5m",
    "roll15_net_bikes",
    "roll30_net_bikes",
    "roll60_net_bikes",
    "roll15_bikes_mean",
    "roll30_bikes_mean",
    "roll60_bikes_mean",
    "neighbor_count_within_radius",
]
WEATHER_PSI_FEATURE_COLUMNS = [
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms",
    "precipitation_mm",
    "hourly_temperature_c",
    "hourly_humidity_pct",
    "hourly_wind_speed_ms",
    "hourly_precipitation_mm",
    "hourly_precipitation_probability_pct",
]
PSI_FEATURE_GROUPS: dict[str, list[str]] = {
    "core": CORE_PSI_FEATURE_COLUMNS,
    "weather": WEATHER_PSI_FEATURE_COLUMNS,
}
PSI_FEATURE_COLUMNS = CORE_PSI_FEATURE_COLUMNS + WEATHER_PSI_FEATURE_COLUMNS
PSI_AGGREGATORS = ("max", "mean", "trimmed_mean", "p75")
DEFAULT_PSI_AGGREGATOR = "trimmed_mean"
TRIMMED_MEAN_RATIO = 0.10
DT_FORMAT = "%Y-%m-%d-%H-%M"


@dataclass(frozen=True)
class DriftWindow:
    baseline_start: str
    baseline_end: str
    recent_start: str
    recent_end: str


def _format_dt(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime(DT_FORMAT)


def build_drift_window(
    *,
    now_utc: datetime,
    lookback_hours: int = 24,
    baseline_days: int = 7,
) -> DriftWindow:
    recent_end = now_utc.astimezone(timezone.utc).replace(second=0, microsecond=0)
    recent_start = recent_end - timedelta(hours=lookback_hours)
    baseline_end = recent_start
    baseline_start = baseline_end - timedelta(days=baseline_days)
    return DriftWindow(
        baseline_start=_format_dt(baseline_start),
        baseline_end=_format_dt(baseline_end),
        recent_start=_format_dt(recent_start),
        recent_end=_format_dt(recent_end),
    )


def load_latest_feature_dt(*, config: PostgresFeatureConfig, city: str) -> datetime:
    engine = create_pg_engine(config)
    try:
        with engine.connect() as connection:
            return load_latest_feature_dt_from_connection(connection=connection, config=config, city=city)
    finally:
        engine.dispose()


def load_latest_feature_dt_from_connection(*, connection, config: PostgresFeatureConfig, city: str) -> datetime:
    sql = text(
        f"""
        SELECT max(dt) AS max_dt
        FROM "{validate_identifier(config.pg_schema)}"."{validate_identifier(config.training_table)}"
        WHERE city = :city
        """
    )
    max_dt = pd.read_sql_query(sql, connection, params={"city": city}).iloc[0]["max_dt"]
    if not max_dt:
        raise RuntimeError(f"feature freshness check failed: no features found for city={city}")
    return datetime.strptime(max_dt, DT_FORMAT).replace(tzinfo=timezone.utc)


def load_feature_window_row_count(
    *,
    connection,
    config: PostgresFeatureConfig,
    city: str,
    start_dt: str,
    end_dt: str,
) -> int:
    sql = text(
        f"""
        SELECT count(*) AS row_count
        FROM "{validate_identifier(config.pg_schema)}"."{validate_identifier(config.training_table)}"
        WHERE city = :city
          AND dt >= :start_dt
          AND dt < :end_dt
        """
    )
    row_count = connection.execute(
        sql,
        {"city": city, "start_dt": start_dt, "end_dt": end_dt},
    ).scalar_one()
    return int(row_count or 0)


def load_feature_series(
    *,
    connection,
    config: PostgresFeatureConfig,
    city: str,
    start_dt: str,
    end_dt: str,
    feature_column: str,
) -> pd.Series:
    if feature_column not in FEATURE_COLUMNS:
        raise ValueError(f"unsupported PSI feature column: {feature_column}")

    sql = text(
        f"""
        SELECT "{validate_identifier(feature_column)}" AS feature_value
        FROM "{validate_identifier(config.pg_schema)}"."{validate_identifier(config.training_table)}"
        WHERE city = :city
          AND dt >= :start_dt
          AND dt < :end_dt
          AND "{validate_identifier(feature_column)}" IS NOT NULL
        """
    )
    return pd.read_sql_query(
        sql,
        connection,
        params={"city": city, "start_dt": start_dt, "end_dt": end_dt},
    )["feature_value"]


def assert_feature_freshness(
    *,
    latest_feature_dt: datetime,
    max_feature_age_minutes: int,
    now_utc: datetime | None = None,
) -> timedelta:
    reference_now = now_utc or datetime.now(timezone.utc)
    age = reference_now.astimezone(timezone.utc) - latest_feature_dt.astimezone(timezone.utc)
    if age > timedelta(minutes=max_feature_age_minutes):
        raise RuntimeError(
            "feature freshness check failed: "
            f"latest dt {latest_feature_dt.isoformat()} age={age} exceeds {max_feature_age_minutes} minutes"
        )
    return age


def load_feature_window(
    *,
    config: PostgresFeatureConfig,
    city: str,
    start_dt: str,
    end_dt: str,
    feature_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    selected_columns = [column for column in (feature_columns or PSI_FEATURE_COLUMNS) if column in FEATURE_COLUMNS]
    if not selected_columns:
        raise ValueError("no PSI feature columns selected")

    select_list = ", ".join(f'"{column}" AS "{column}"' for column in selected_columns)
    sql = text(
        f"""
        SELECT {select_list}
        FROM "{validate_identifier(config.pg_schema)}"."{validate_identifier(config.training_table)}"
        WHERE city = :city
          AND dt >= :start_dt
          AND dt < :end_dt
        """
    )
    engine = create_pg_engine(config)
    try:
        with engine.connect() as connection:
            return pd.read_sql_query(
                sql,
                connection,
                params={"city": city, "start_dt": start_dt, "end_dt": end_dt},
            )
    finally:
        engine.dispose()


def _psi_component(expected: float, actual: float, *, epsilon: float = 1e-6) -> float:
    expected = max(expected, epsilon)
    actual = max(actual, epsilon)
    return (actual - expected) * np.log(actual / expected)


def _psi_from_numeric_series(
    baseline: pd.Series,
    recent: pd.Series,
    *,
    bins: int = 10,
) -> float:
    baseline = pd.to_numeric(baseline, errors="coerce").dropna()
    recent = pd.to_numeric(recent, errors="coerce").dropna()
    if baseline.empty or recent.empty:
        return 0.0

    quantiles = np.unique(np.quantile(baseline.to_numpy(dtype=float), np.linspace(0.0, 1.0, bins + 1)))
    if quantiles.size < 2:
        return 0.0

    bin_edges = quantiles.copy()
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    baseline_hist, _ = np.histogram(baseline.to_numpy(dtype=float), bins=bin_edges)
    recent_hist, _ = np.histogram(recent.to_numpy(dtype=float), bins=bin_edges)
    baseline_props = baseline_hist / max(1, baseline_hist.sum())
    recent_props = recent_hist / max(1, recent_hist.sum())
    return float(sum(_psi_component(expected, actual) for expected, actual in zip(baseline_props, recent_props)))


def compute_feature_psi_map(
    baseline_df: pd.DataFrame,
    recent_df: pd.DataFrame,
    *,
    feature_columns: Iterable[str] | None = None,
) -> dict[str, float]:
    selected_columns = [column for column in (feature_columns or PSI_FEATURE_COLUMNS) if column in baseline_df.columns]
    if not selected_columns:
        raise ValueError("no overlapping PSI feature columns available")
    return {
        column: _psi_from_numeric_series(baseline_df[column], recent_df[column])
        for column in selected_columns
        if column in recent_df.columns
    }


def compute_feature_psi_map_from_db(
    *,
    connection,
    config: PostgresFeatureConfig,
    city: str,
    baseline_start_dt: str,
    baseline_end_dt: str,
    recent_start_dt: str,
    recent_end_dt: str,
    feature_columns: Iterable[str] | None = None,
) -> dict[str, float]:
    selected_columns = [column for column in (feature_columns or PSI_FEATURE_COLUMNS) if column in FEATURE_COLUMNS]
    if not selected_columns:
        raise ValueError("no PSI feature columns selected")

    feature_psis: dict[str, float] = {}
    for column in selected_columns:
        baseline_series = load_feature_series(
            connection=connection,
            config=config,
            city=city,
            start_dt=baseline_start_dt,
            end_dt=baseline_end_dt,
            feature_column=column,
        )
        recent_series = load_feature_series(
            connection=connection,
            config=config,
            city=city,
            start_dt=recent_start_dt,
            end_dt=recent_end_dt,
            feature_column=column,
        )
        feature_psis[column] = _psi_from_numeric_series(baseline_series, recent_series)
    return feature_psis


def split_feature_psi_groups(feature_psis: dict[str, float]) -> dict[str, dict[str, float]]:
    return {
        group_name: {column: feature_psis[column] for column in group_columns if column in feature_psis}
        for group_name, group_columns in PSI_FEATURE_GROUPS.items()
    }


def _trimmed_mean(values: np.ndarray, *, trim_ratio: float = TRIMMED_MEAN_RATIO) -> float:
    if values.size < 5:
        return float(np.mean(values))
    trim_count = max(1, int(round(values.size * trim_ratio)))
    trim_count = min(trim_count, (values.size - 1) // 2)
    if trim_count <= 0:
        return float(np.mean(values))
    trimmed = values[trim_count : values.size - trim_count]
    return float(np.mean(trimmed))


def aggregate_psi(feature_psis: dict[str, float], *, aggregator: str = DEFAULT_PSI_AGGREGATOR) -> float:
    if not feature_psis:
        return 0.0
    values = np.sort(np.asarray(list(feature_psis.values()), dtype=float))
    if aggregator == "mean":
        return float(np.mean(values))
    if aggregator == "trimmed_mean":
        return _trimmed_mean(values)
    if aggregator == "p75":
        return float(np.percentile(values, 75))
    if aggregator == "max":
        return float(np.max(values))
    raise ValueError(f"unsupported PSI aggregator: {aggregator}")


def publish_psi(
    *,
    config: PostgresFeatureConfig,
    city: str,
    endpoint: str,
    target_name: str,
    environment: str,
    lookback_hours: int = 24,
    baseline_days: int = 7,
    aggregator: str = DEFAULT_PSI_AGGREGATOR,
    max_feature_age_minutes: int = 45,
    dry_run: bool = False,
) -> dict[str, object]:
    result = compute_psi_result(
        config=config,
        city=city,
        lookback_hours=lookback_hours,
        baseline_days=baseline_days,
        aggregator=aggregator,
        max_feature_age_minutes=max_feature_age_minutes,
    )
    if dry_run:
        return result
    publish_psi_metrics(
        result,
        endpoint=endpoint,
        city=city,
        target_name=target_name,
        environment=environment,
    )
    return result


def compute_psi_result(
    *,
    config: PostgresFeatureConfig,
    city: str,
    lookback_hours: int = 24,
    baseline_days: int = 7,
    aggregator: str = DEFAULT_PSI_AGGREGATOR,
    max_feature_age_minutes: int = 45,
) -> dict[str, object]:
    engine = create_pg_engine(config)
    try:
        with engine.connect() as connection:
            latest_feature_dt = load_latest_feature_dt_from_connection(connection=connection, config=config, city=city)
            feature_age = assert_feature_freshness(
                latest_feature_dt=latest_feature_dt,
                max_feature_age_minutes=max_feature_age_minutes,
            )
            window = build_drift_window(
                now_utc=datetime.now(timezone.utc),
                lookback_hours=lookback_hours,
                baseline_days=baseline_days,
            )
            baseline_rows = load_feature_window_row_count(
                connection=connection,
                config=config,
                city=city,
                start_dt=window.baseline_start,
                end_dt=window.baseline_end,
            )
            recent_rows = load_feature_window_row_count(
                connection=connection,
                config=config,
                city=city,
                start_dt=window.recent_start,
                end_dt=window.recent_end,
            )
            print(
                f"[psi] city={city} baseline_rows={baseline_rows} recent_rows={recent_rows} "
                f"baseline={window.baseline_start}..{window.baseline_end} "
                f"recent={window.recent_start}..{window.recent_end}"
            )
            if recent_rows == 0:
                raise RuntimeError(
                    "feature freshness check failed: "
                    f"recent feature window is empty for city={city}, window={window.recent_start}..{window.recent_end}"
                )

            if baseline_rows == 0 or recent_rows == 0:
                result = {
                    "psi": 0.0,
                    "psi_core": 0.0,
                    "psi_weather": 0.0,
                    "psi_aggregator": aggregator,
                    "feature_count": 0,
                    "core_feature_count": 0,
                    "weather_feature_count": 0,
                    "baseline_rows": baseline_rows,
                    "recent_rows": recent_rows,
                    "latest_feature_dt_utc": latest_feature_dt.isoformat(),
                    "feature_age_minutes": round(feature_age.total_seconds() / 60.0, 3),
                    "window": window,
                }
                return result

            # Compute one feature at a time so the tier-2 worker does not materialize
            # multi-day baseline and recent windows for every PSI column at once.
            feature_psis = compute_feature_psi_map_from_db(
                connection=connection,
                config=config,
                city=city,
                baseline_start_dt=window.baseline_start,
                baseline_end_dt=window.baseline_end,
                recent_start_dt=window.recent_start,
                recent_end_dt=window.recent_end,
            )
    finally:
        engine.dispose()

    grouped_feature_psis = split_feature_psi_groups(feature_psis)
    psi_value = aggregate_psi(feature_psis, aggregator=aggregator)
    psi_core = aggregate_psi(grouped_feature_psis["core"], aggregator=aggregator)
    psi_weather = aggregate_psi(grouped_feature_psis["weather"], aggregator=aggregator)
    result = {
        "psi": psi_value,
        "psi_core": psi_core,
        "psi_weather": psi_weather,
        "psi_aggregator": aggregator,
        "feature_count": len(feature_psis),
        "core_feature_count": len(grouped_feature_psis["core"]),
        "weather_feature_count": len(grouped_feature_psis["weather"]),
        "baseline_rows": baseline_rows,
        "recent_rows": recent_rows,
        "latest_feature_dt_utc": latest_feature_dt.isoformat(),
        "feature_age_minutes": round(feature_age.total_seconds() / 60.0, 3),
        "window": window,
        "top_features": sorted(feature_psis.items(), key=lambda item: item[1], reverse=True)[:5],
        "top_core_features": sorted(grouped_feature_psis["core"].items(), key=lambda item: item[1], reverse=True)[:5],
        "top_weather_features": sorted(grouped_feature_psis["weather"].items(), key=lambda item: item[1], reverse=True)[
            :5
        ],
    }
    return result


def publish_psi_metrics(
    result: dict[str, object],
    *,
    endpoint: str,
    city: str,
    target_name: str,
    environment: str,
) -> None:
    put_metrics_bulk(
        [
            ("PSI", float(result["psi"]), "None"),
            ("PSI_core", float(result["psi_core"]), "None"),
            ("PSI_weather", float(result["psi_weather"]), "None"),
        ],
        endpoint=endpoint,
        city=city,
        target_name=target_name,
        environment=environment,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and publish target-aware PSI from Postgres feature history.")
    parser.add_argument("--city", default="paris")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--target-name", required=True, choices=["bikes", "docks"])
    parser.add_argument("--environment", default="production")
    parser.add_argument("--pg-host", required=True)
    parser.add_argument("--pg-port", type=int, default=5432)
    parser.add_argument("--pg-db", required=True)
    parser.add_argument("--pg-user", required=True)
    parser.add_argument("--pg-password", required=True)
    parser.add_argument("--pg-schema", default="analytics")
    parser.add_argument("--feature-table", default="feat_station_snapshot_5min")
    parser.add_argument("--lookback-hours", type=int, default=24)
    parser.add_argument("--baseline-days", type=int, default=7)
    parser.add_argument("--aggregator", choices=PSI_AGGREGATORS, default=DEFAULT_PSI_AGGREGATOR)
    parser.add_argument(
        "--max-feature-age-minutes",
        type=int,
        default=int(os.environ.get("PSI_MAX_FEATURE_AGE_MINUTES", "45")),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    result = publish_psi(
        config=PostgresFeatureConfig(
            pg_host=args.pg_host,
            pg_port=args.pg_port,
            pg_db=args.pg_db,
            pg_user=args.pg_user,
            pg_password=args.pg_password,
            pg_schema=args.pg_schema,
            training_table=args.feature_table,
        ),
        city=args.city,
        endpoint=args.endpoint,
        target_name=args.target_name,
        environment=args.environment,
        lookback_hours=args.lookback_hours,
        baseline_days=args.baseline_days,
        aggregator=args.aggregator,
        max_feature_age_minutes=args.max_feature_age_minutes,
        dry_run=args.dry_run,
    )
    print(result)
    return result


if __name__ == "__main__":
    main()
