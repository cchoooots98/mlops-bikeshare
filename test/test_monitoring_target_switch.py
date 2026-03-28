from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from src.config import quality_key
from src.model_target import target_spec_from_predict_bikes
from src.monitoring import quality_backfill
from src.monitoring.metrics import publish_custom_metrics, publish_psi, publish_psi_all_targets
from src.monitoring.metrics.metrics_helper import build_metric_dimensions


class _FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeEngine:
    def __init__(self):
        self.disposed = False

    def connect(self):
        return _FakeConnection()

    def dispose(self):
        self.disposed = True


def test_quality_backfill_builds_docks_rows():
    target_spec = target_spec_from_predict_bikes(False)
    preds = pd.DataFrame(
        [
            {
                "station_id": "station-001",
                "prediction_target": "docks",
                "yhat_docks": 0.42,
                "yhat_docks_bin": 1.0,
                "inferenceId": "2026-03-11-10-05_station-001",
            }
        ]
    )
    acts = pd.DataFrame(
        [
            {
                "station_id": "station-001",
                "dt_plus30": "2026-03-11-10-35",
                "min_inventory_within_30m": pd.NA,
                "docks_t30": 15,
                "y_stockout_docks_30": 1,
            }
        ]
    )

    joined = quality_backfill._build_quality_rows(preds, acts, "2026-03-11-10-05", target_spec)

    assert list(joined.columns) == [
        "station_id",
        "dt",
        "dt_plus30",
        "prediction_target",
        "yhat_docks",
        "yhat_docks_bin",
        "y_stockout_docks_30",
        "min_inventory_within_30m",
        "docks_t30",
        "inferenceId",
    ]


def test_publish_custom_metrics_resolves_docks_columns():
    df = pd.DataFrame(
        [
            {
                "prediction_target": "docks",
                "yhat_docks": 0.7,
                "y_stockout_docks_30": 1,
            }
        ]
    )

    assert publish_custom_metrics.resolve_quality_columns(df) == ("y_stockout_docks_30", "yhat_docks")


def test_quality_key_is_target_partitioned():
    assert quality_key("paris", "2026-03-11-10-05", "docks") == (
        "monitoring/quality/target=docks/city=paris/ds=2026-03-11/part-2026-03-11-10-05.parquet"
    )


def test_metric_dimensions_include_environment_and_target():
    dims = build_metric_dimensions(
        endpoint="bikeshare-docks-prod",
        city="paris",
        target_name="docks",
        environment="production",
    )

    assert dims == [
        {"Name": "Environment", "Value": "production"},
        {"Name": "EndpointName", "Value": "bikeshare-docks-prod"},
        {"Name": "City", "Value": "paris"},
        {"Name": "TargetName", "Value": "docks"},
    ]


def test_publish_psi_computes_target_aware_feature_map():
    baseline = pd.DataFrame(
        {
            "util_bikes": [0.10, 0.15, 0.20, 0.22],
            "temperature_c": [10.0, 11.0, 12.0, 12.5],
        }
    )
    recent = pd.DataFrame(
        {
            "util_bikes": [0.35, 0.40, 0.45, 0.50],
            "temperature_c": [14.0, 15.0, 16.0, 17.0],
        }
    )

    feature_psis = publish_psi.compute_feature_psi_map(
        baseline,
        recent,
        feature_columns=["util_bikes", "temperature_c"],
    )

    assert set(feature_psis) == {"util_bikes", "temperature_c"}
    assert feature_psis["util_bikes"] > 0
    assert publish_psi.aggregate_psi(feature_psis, aggregator="max") >= feature_psis["temperature_c"]


def test_publish_psi_supports_robust_aggregators_and_groups():
    feature_psis = {
        "util_bikes": 0.10,
        "util_docks": 0.20,
        "delta_bikes_5m": 0.30,
        "delta_docks_5m": 0.40,
        "roll15_net_bikes": 0.50,
        "roll30_net_bikes": 0.60,
        "roll60_net_bikes": 0.70,
        "roll15_bikes_mean": 0.80,
        "roll30_bikes_mean": 0.90,
        "roll60_bikes_mean": 1.00,
        "neighbor_count_within_radius": 4.00,
        "temperature_c": 0.15,
        "humidity_pct": 0.25,
        "wind_speed_ms": 0.35,
        "precipitation_mm": 0.45,
        "hourly_temperature_c": 0.55,
        "hourly_humidity_pct": 0.65,
        "hourly_wind_speed_ms": 0.75,
        "hourly_precipitation_mm": 0.85,
        "hourly_precipitation_probability_pct": 3.50,
    }

    groups = publish_psi.split_feature_psi_groups(feature_psis)

    assert set(groups) == {"core", "weather"}
    assert set(groups["core"]) == set(publish_psi.CORE_PSI_FEATURE_COLUMNS)
    assert set(groups["weather"]) == set(publish_psi.WEATHER_PSI_FEATURE_COLUMNS)
    assert publish_psi.aggregate_psi(feature_psis, aggregator="max") == 4.0
    assert publish_psi.aggregate_psi(feature_psis, aggregator="p75") > 0
    assert publish_psi.aggregate_psi(feature_psis, aggregator="p75") < publish_psi.aggregate_psi(
        feature_psis, aggregator="max"
    )
    assert publish_psi.aggregate_psi(feature_psis, aggregator="trimmed_mean") < publish_psi.aggregate_psi(
        feature_psis, aggregator="max"
    )
    assert publish_psi.aggregate_psi(groups["core"], aggregator="trimmed_mean") > 0
    assert publish_psi.aggregate_psi(groups["weather"], aggregator="trimmed_mean") > 0


def test_publish_psi_feature_freshness_rejects_stale_latest_dt():
    latest_dt = datetime(2026, 3, 18, 0, 0, tzinfo=timezone.utc)

    with pytest.raises(RuntimeError, match="feature freshness check failed"):
        publish_psi.assert_feature_freshness(
            latest_feature_dt=latest_dt,
            max_feature_age_minutes=45,
            now_utc=datetime(2026, 3, 18, 1, 0, tzinfo=timezone.utc),
        )


def test_publish_psi_propagates_compute_errors_without_publishing(monkeypatch):
    config = publish_psi.PostgresFeatureConfig(
        pg_host="dw-postgres",
        pg_port=5432,
        pg_db="velib_dw",
        pg_user="velib",
        pg_password="velib",
    )
    publish_calls = []
    monkeypatch.setattr(
        publish_psi,
        "compute_psi_result",
        lambda **_: (_ for _ in ()).throw(RuntimeError("recent feature window is empty")),
    )
    monkeypatch.setattr(
        publish_psi,
        "publish_psi_metrics",
        lambda *args, **kwargs: publish_calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="recent feature window is empty"):
        publish_psi.publish_psi(
            config=config,
            city="paris",
            endpoint="bikeshare-bikes-staging",
            target_name="bikes",
            environment="staging",
            dry_run=True,
        )

    assert publish_calls == []


def test_publish_psi_dry_run_returns_compute_result_without_publishing(monkeypatch):
    config = publish_psi.PostgresFeatureConfig(
        pg_host="dw-postgres",
        pg_port=5432,
        pg_db="velib_dw",
        pg_user="velib",
        pg_password="velib",
    )
    compute_calls = []
    publish_calls = []
    monkeypatch.setattr(
        publish_psi,
        "compute_psi_result",
        lambda **kwargs: compute_calls.append(kwargs)
        or {
            "psi": 0.22,
            "psi_core": 0.31,
            "psi_weather": 0.18,
            "psi_aggregator": "trimmed_mean",
            "feature_count": 3,
            "core_feature_count": 1,
            "weather_feature_count": 2,
            "top_core_features": [("util_bikes", 0.31)],
            "top_weather_features": [("temperature_c", 0.18), ("wind_speed_ms", 0.12)],
        },
    )
    monkeypatch.setattr(
        publish_psi,
        "publish_psi_metrics",
        lambda *args, **kwargs: publish_calls.append((args, kwargs)),
    )

    result = publish_psi.publish_psi(
        config=config,
        city="paris",
        endpoint="bikeshare-bikes-staging",
        target_name="bikes",
        environment="staging",
        aggregator="trimmed_mean",
        dry_run=True,
    )

    assert result["psi_aggregator"] == "trimmed_mean"
    assert result["psi"] > 0
    assert result["psi_core"] > 0
    assert result["psi_weather"] > 0
    assert result["core_feature_count"] == 1
    assert result["weather_feature_count"] == 2
    assert result["top_core_features"][0][0] == "util_bikes"
    assert result["top_weather_features"][0][0] in {"temperature_c", "wind_speed_ms"}
    assert compute_calls == [
        {
            "config": config,
            "city": "paris",
            "lookback_hours": 24,
            "baseline_days": 7,
            "aggregator": "trimmed_mean",
            "max_feature_age_minutes": 45,
            "query_chunk_size": publish_psi.DEFAULT_PSI_QUERY_CHUNK_SIZE,
        }
    ]
    assert publish_calls == []


def test_publish_psi_publishes_metrics_for_single_target(monkeypatch):
    config = publish_psi.PostgresFeatureConfig(
        pg_host="dw-postgres",
        pg_port=5432,
        pg_db="velib_dw",
        pg_user="velib",
        pg_password="velib",
    )
    compute_result = {
        "psi": 0.27,
        "psi_core": 0.19,
        "psi_weather": 0.11,
        "psi_aggregator": "trimmed_mean",
        "feature_count": 2,
        "core_feature_count": 1,
        "weather_feature_count": 1,
    }
    publish_calls = []

    monkeypatch.setattr(publish_psi, "compute_psi_result", lambda **_: compute_result)
    monkeypatch.setattr(
        publish_psi,
        "publish_psi_metrics",
        lambda *args, **kwargs: publish_calls.append((args, kwargs)),
    )

    result = publish_psi.publish_psi(
        config=config,
        city="paris",
        endpoint="bikeshare-bikes-prod",
        target_name="bikes",
        environment="production",
        dry_run=False,
    )

    assert result == compute_result
    assert publish_calls == [
        (
            (compute_result,),
            {
                "endpoint": "bikeshare-bikes-prod",
                "city": "paris",
                "target_name": "bikes",
                "environment": "production",
            },
        )
    ]


def test_compute_feature_psi_map_from_db_loads_feature_windows_in_chunks(monkeypatch):
    config = publish_psi.PostgresFeatureConfig(
        pg_host="dw-postgres",
        pg_port=5432,
        pg_db="velib_dw",
        pg_user="velib",
        pg_password="velib",
    )

    window_calls: list[tuple[tuple[str, ...], str, str]] = []

    def fake_load_feature_window_from_connection(**kwargs):
        chunk = tuple(kwargs["feature_columns"])
        start_dt = kwargs["start_dt"]
        end_dt = kwargs["end_dt"]
        window_calls.append((chunk, start_dt, end_dt))
        if start_dt == "baseline-start":
            values = {
                "util_bikes": [0.10, 0.15, 0.20],
                "temperature_c": [10.0, 11.0, 12.0],
            }
        else:
            values = {
                "util_bikes": [0.30, 0.35, 0.40],
                "temperature_c": [14.0, 15.0, 16.0],
            }
        return pd.DataFrame({column: values[column] for column in chunk})

    monkeypatch.setattr(
        publish_psi,
        "load_feature_window_from_connection",
        fake_load_feature_window_from_connection,
    )

    feature_psis = publish_psi.compute_feature_psi_map_from_db(
        connection=object(),
        config=config,
        city="paris",
        baseline_start_dt="baseline-start",
        baseline_end_dt="baseline-end",
        recent_start_dt="recent-start",
        recent_end_dt="recent-end",
        feature_columns=["util_bikes", "temperature_c"],
        query_chunk_size=1,
    )

    assert set(feature_psis) == {"util_bikes", "temperature_c"}
    assert feature_psis["util_bikes"] > 0
    assert feature_psis["temperature_c"] > 0
    assert window_calls == [
        (("util_bikes",), "baseline-start", "baseline-end"),
        (("util_bikes",), "recent-start", "recent-end"),
        (("temperature_c",), "baseline-start", "baseline-end"),
        (("temperature_c",), "recent-start", "recent-end"),
    ]


def test_compute_psi_result_raises_when_recent_feature_window_is_empty(monkeypatch):
    config = publish_psi.PostgresFeatureConfig(
        pg_host="dw-postgres",
        pg_port=5432,
        pg_db="velib_dw",
        pg_user="velib",
        pg_password="velib",
    )
    fake_engine = _FakeEngine()

    monkeypatch.setattr(publish_psi, "create_pg_engine", lambda _: fake_engine)
    monkeypatch.setattr(
        publish_psi,
        "load_latest_feature_dt_from_connection",
        lambda **_: datetime(2026, 3, 18, 0, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        publish_psi,
        "assert_feature_freshness",
        lambda **_: timedelta(minutes=6),
    )
    monkeypatch.setattr(
        publish_psi,
        "build_drift_window",
        lambda **_: publish_psi.DriftWindow(
            baseline_start="2026-03-10-00-00",
            baseline_end="2026-03-17-00-00",
            recent_start="2026-03-17-00-00",
            recent_end="2026-03-18-00-00",
        ),
    )

    row_counts = iter((12, 0))
    monkeypatch.setattr(
        publish_psi,
        "load_feature_window_row_count",
        lambda **_: next(row_counts),
    )

    with pytest.raises(RuntimeError, match="recent feature window is empty"):
        publish_psi.compute_psi_result(config=config, city="paris")

    assert fake_engine.disposed is True


def test_compute_psi_result_aggregates_feature_map_from_db(monkeypatch):
    config = publish_psi.PostgresFeatureConfig(
        pg_host="dw-postgres",
        pg_port=5432,
        pg_db="velib_dw",
        pg_user="velib",
        pg_password="velib",
    )
    fake_engine = _FakeEngine()

    monkeypatch.setattr(publish_psi, "create_pg_engine", lambda _: fake_engine)
    monkeypatch.setattr(
        publish_psi,
        "load_latest_feature_dt_from_connection",
        lambda **_: datetime(2026, 3, 18, 0, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        publish_psi,
        "assert_feature_freshness",
        lambda **_: timedelta(minutes=4),
    )
    monkeypatch.setattr(
        publish_psi,
        "build_drift_window",
        lambda **_: publish_psi.DriftWindow(
            baseline_start="2026-03-10-00-00",
            baseline_end="2026-03-17-00-00",
            recent_start="2026-03-17-00-00",
            recent_end="2026-03-18-00-00",
        ),
    )

    row_counts = iter((48, 16))
    monkeypatch.setattr(
        publish_psi,
        "load_feature_window_row_count",
        lambda **_: next(row_counts),
    )
    monkeypatch.setattr(
        publish_psi,
        "compute_feature_psi_map_from_db",
        lambda **_: {
            "util_bikes": 0.45,
            "temperature_c": 0.15,
        },
    )

    result = publish_psi.compute_psi_result(
        config=config,
        city="paris",
        aggregator="trimmed_mean",
    )

    assert result["psi"] > 0
    assert result["psi_core"] == 0.45
    assert result["psi_weather"] == 0.15
    assert result["baseline_rows"] == 48
    assert result["recent_rows"] == 16
    assert result["feature_count"] == 2
    assert result["core_feature_count"] == 1
    assert result["weather_feature_count"] == 1
    assert result["top_features"][0][0] == "util_bikes"
    assert fake_engine.disposed is True


def test_publish_psi_all_targets_dry_run_returns_targets_without_publishing(monkeypatch):
    monkeypatch.setattr(
        publish_psi_all_targets,
        "compute_psi_result",
        lambda **_: {
            "psi": 0.21,
            "psi_core": 0.18,
            "psi_weather": 0.09,
            "psi_aggregator": "trimmed_mean",
            "feature_count": 2,
            "core_feature_count": 1,
            "weather_feature_count": 1,
        },
    )
    publish_calls = []
    monkeypatch.setattr(
        publish_psi_all_targets,
        "publish_psi_metrics",
        lambda *args, **kwargs: publish_calls.append((args, kwargs)),
    )

    payload = publish_psi_all_targets.main(
        [
            "--city",
            "paris",
            "--environment",
            "staging",
            "--pg-host",
            "dw-postgres",
            "--pg-db",
            "velib_dw",
            "--pg-user",
            "velib",
            "--pg-password",
            "velib",
            "--dry-run",
        ]
    )

    assert [target["target_name"] for target in payload["targets"]] == ["bikes", "docks"]
    assert [target["endpoint"] for target in payload["targets"]] == [
        "bikeshare-bikes-staging",
        "bikeshare-docks-staging",
    ]
    assert publish_calls == []


def test_publish_psi_all_targets_publishes_once_per_target(monkeypatch):
    monkeypatch.setattr(
        publish_psi_all_targets,
        "compute_psi_result",
        lambda **_: {
            "psi": 0.31,
            "psi_core": 0.24,
            "psi_weather": 0.12,
            "psi_aggregator": "trimmed_mean",
            "feature_count": 2,
            "core_feature_count": 1,
            "weather_feature_count": 1,
        },
    )
    publish_calls = []

    def fake_publish(result, **kwargs):
        publish_calls.append((result, kwargs))

    monkeypatch.setattr(publish_psi_all_targets, "publish_psi_metrics", fake_publish)

    payload = publish_psi_all_targets.main(
        [
            "--city",
            "paris",
            "--environment",
            "production",
            "--pg-host",
            "dw-postgres",
            "--pg-db",
            "velib_dw",
            "--pg-user",
            "velib",
            "--pg-password",
            "velib",
        ]
    )

    assert len(publish_calls) == 2
    assert [call[1]["target_name"] for call in publish_calls] == ["bikes", "docks"]
    assert [call[1]["endpoint"] for call in publish_calls] == [
        "bikeshare-bikes-prod",
        "bikeshare-docks-prod",
    ]
    assert payload["environment"] == "production"
