import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.features.postgres_store import PostgresFeatureConfig, create_pg_engine, validate_identifier
from src.features.schema import FEATURE_COLUMNS
from src.monitoring.metrics.metrics_helper import put_metric


PSI_FEATURE_COLUMNS = [
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


def aggregate_psi(feature_psis: dict[str, float], *, aggregator: str = "max") -> float:
    if not feature_psis:
        return 0.0
    values = list(feature_psis.values())
    if aggregator == "mean":
        return float(np.mean(values))
    return float(max(values))


def publish_psi(
    *,
    config: PostgresFeatureConfig,
    city: str,
    endpoint: str,
    target_name: str,
    environment: str,
    lookback_hours: int = 24,
    baseline_days: int = 7,
    aggregator: str = "max",
    dry_run: bool = False,
) -> dict[str, object]:
    window = build_drift_window(
        now_utc=datetime.now(timezone.utc),
        lookback_hours=lookback_hours,
        baseline_days=baseline_days,
    )
    baseline_df = load_feature_window(
        config=config,
        city=city,
        start_dt=window.baseline_start,
        end_dt=window.baseline_end,
    )
    recent_df = load_feature_window(
        config=config,
        city=city,
        start_dt=window.recent_start,
        end_dt=window.recent_end,
    )

    if baseline_df.empty or recent_df.empty:
        result = {
            "psi": 0.0,
            "feature_count": 0,
            "baseline_rows": int(len(baseline_df)),
            "recent_rows": int(len(recent_df)),
            "window": window,
        }
        if dry_run:
            return result
        put_metric(
            "PSI",
            0.0,
            endpoint=endpoint,
            city=city,
            target_name=target_name,
            environment=environment,
        )
        return result

    feature_psis = compute_feature_psi_map(baseline_df, recent_df)
    psi_value = aggregate_psi(feature_psis, aggregator=aggregator)
    result = {
        "psi": psi_value,
        "feature_count": len(feature_psis),
        "baseline_rows": int(len(baseline_df)),
        "recent_rows": int(len(recent_df)),
        "window": window,
        "top_features": sorted(feature_psis.items(), key=lambda item: item[1], reverse=True)[:5],
    }
    if dry_run:
        return result

    put_metric(
        "PSI",
        psi_value,
        endpoint=endpoint,
        city=city,
        target_name=target_name,
        environment=environment,
    )
    return result


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
    parser.add_argument("--aggregator", choices=["max", "mean"], default="max")
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
        dry_run=args.dry_run,
    )
    print(result)
    return result


if __name__ == "__main__":
    main()
