import pandas as pd

from src.config import quality_key
from src.model_target import target_spec_from_predict_bikes
from src.monitoring import quality_backfill
from src.monitoring.metrics import publish_custom_metrics
from src.monitoring.metrics import publish_psi
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
