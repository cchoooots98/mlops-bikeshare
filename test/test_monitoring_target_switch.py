from datetime import datetime, timezone

import pandas as pd
import pytest
from src.config import quality_key
from src.model_target import target_spec_from_predict_bikes
from src.monitoring import quality_backfill
from src.monitoring.metrics import publish_custom_metrics, publish_psi
from src.monitoring.metrics.metrics_helper import build_metric_dimensions


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


def test_publish_psi_raises_when_recent_feature_window_is_empty(monkeypatch):
    config = publish_psi.PostgresFeatureConfig(
        pg_host="dw-postgres",
        pg_port=5432,
        pg_db="velib_dw",
        pg_user="velib",
        pg_password="velib",
    )

    monkeypatch.setattr(
        publish_psi,
        "load_latest_feature_dt",
        lambda **_: datetime(2026, 3, 18, 0, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        publish_psi,
        "assert_feature_freshness",
        lambda **_: datetime.now(timezone.utc) - datetime(2026, 3, 18, 0, 0, tzinfo=timezone.utc),
    )

    call_count = {"value": 0}

    def fake_load_feature_window(**kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            return pd.DataFrame({"util_bikes": [0.1], "temperature_c": [10.0]})
        return pd.DataFrame()

    monkeypatch.setattr(publish_psi, "load_feature_window", fake_load_feature_window)

    with pytest.raises(RuntimeError, match="recent feature window is empty"):
        publish_psi.publish_psi(
            config=config,
            city="paris",
            endpoint="bikeshare-bikes-staging",
            target_name="bikes",
            environment="staging",
            dry_run=True,
        )


def test_publish_psi_dry_run_returns_core_and_weather_metrics(monkeypatch):
    config = publish_psi.PostgresFeatureConfig(
        pg_host="dw-postgres",
        pg_port=5432,
        pg_db="velib_dw",
        pg_user="velib",
        pg_password="velib",
    )

    monkeypatch.setattr(
        publish_psi,
        "load_latest_feature_dt",
        lambda **_: datetime(2026, 3, 18, 0, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        publish_psi,
        "assert_feature_freshness",
        lambda **_: datetime.now(timezone.utc) - datetime(2026, 3, 18, 0, 0, tzinfo=timezone.utc),
    )

    baseline = pd.DataFrame(
        {
            "util_bikes": [0.10, 0.15, 0.20, 0.22],
            "temperature_c": [10.0, 11.0, 12.0, 12.5],
            "wind_speed_ms": [3.0, 3.5, 4.0, 4.5],
        }
    )
    recent = pd.DataFrame(
        {
            "util_bikes": [0.35, 0.40, 0.45, 0.50],
            "temperature_c": [14.0, 15.0, 16.0, 17.0],
            "wind_speed_ms": [1.5, 1.8, 2.0, 2.2],
        }
    )

    call_count = {"value": 0}

    def fake_load_feature_window(**kwargs):
        call_count["value"] += 1
        return baseline if call_count["value"] == 1 else recent

    monkeypatch.setattr(publish_psi, "load_feature_window", fake_load_feature_window)

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
