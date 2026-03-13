import pandas as pd
import pytest

from src.features.postgres_store import (
    PostgresFeatureConfig,
    build_feature_select_query,
    create_pg_engine,
    load_latest_serving_features,
    load_training_actuals_for_dt,
    load_training_slice,
)
from src.model_target import target_spec_from_predict_bikes
from src.features.schema import ONLINE_REQUIRED_COLUMNS, TRAINING_REQUIRED_COLUMNS


class _FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeEngine:
    def connect(self):
        return _FakeConnection()


def _training_df() -> pd.DataFrame:
    row = {
        "city": "paris",
        "dt": "2026-03-11-10-05",
        "station_id": "station-001",
        "capacity": 40,
        "lat": 48.8566,
        "lon": 2.3522,
        "bikes": 12,
        "docks": 28,
        "minutes_since_prev_snapshot": 5.0,
        "util_bikes": 0.3,
        "util_docks": 0.7,
        "delta_bikes_5m": 1.0,
        "delta_docks_5m": -1.0,
        "roll15_net_bikes": 2.0,
        "roll30_net_bikes": 3.0,
        "roll60_net_bikes": 4.0,
        "roll15_bikes_mean": 10.0,
        "roll30_bikes_mean": 11.0,
        "roll60_bikes_mean": 12.0,
        "nbr_bikes_weighted": 9.5,
        "nbr_docks_weighted": 18.5,
        "has_neighbors_within_radius": 1,
        "neighbor_count_within_radius": 4,
        "hour": 10,
        "dow": 2,
        "is_weekend": 0,
        "is_holiday": 0,
        "temperature_c": 16.2,
        "humidity_pct": 55.0,
        "wind_speed_ms": 4.8,
        "precipitation_mm": 0.2,
        "weather_code": 3,
        "hourly_temperature_c": 16.0,
        "hourly_humidity_pct": 56.0,
        "hourly_wind_speed_ms": 5.1,
        "hourly_precipitation_mm": 0.3,
        "hourly_precipitation_probability_pct": 35.0,
        "hourly_weather_code": 3,
        "y_stockout_bikes_30": 0,
        "y_stockout_docks_30": 0,
        "target_bikes_t30": 10,
        "target_docks_t30": 30,
    }
    return pd.DataFrame([row], columns=TRAINING_REQUIRED_COLUMNS)


def _online_df() -> pd.DataFrame:
    return _training_df()[ONLINE_REQUIRED_COLUMNS].copy()


def test_build_feature_select_query_quotes_schema_table_and_columns():
    sql = build_feature_select_query(
        "analytics",
        "feat_station_snapshot_latest",
        ["city", "dt", "station_id"],
        "city = :city",
        "ORDER BY dt, station_id",
    )

    assert 'FROM "analytics"."feat_station_snapshot_latest"' in sql
    assert '"city" AS "city", "dt" AS "dt", "station_id" AS "station_id"' in sql
    assert "WHERE city = :city ORDER BY dt, station_id" in sql


def test_load_training_slice_reads_training_table(monkeypatch):
    captured = {}

    def fake_read_sql_query(sql, connection, params):
        captured["sql"] = str(sql)
        captured["params"] = params
        return _training_df()

    monkeypatch.setattr(pd, "read_sql_query", fake_read_sql_query)

    config = PostgresFeatureConfig("localhost", 5432, "velib_dw", "velib", "velib")
    df = load_training_slice(_FakeEngine(), config, "paris", "2026-03-01-00-00", "2026-03-02-00-00")

    assert not df.empty
    assert 'FROM "analytics"."feat_station_snapshot_5min"' in captured["sql"]
    assert captured["params"]["city"] == "paris"


def test_load_latest_serving_features_rejects_label_columns(monkeypatch):
    def fake_read_sql_query(sql, connection, params):
        df = _online_df()
        df["y_stockout_bikes_30"] = 0
        return df

    monkeypatch.setattr(pd, "read_sql_query", fake_read_sql_query)

    config = PostgresFeatureConfig("localhost", 5432, "velib_dw", "velib", "velib")
    with pytest.raises(ValueError, match="must not expose label columns"):
        load_latest_serving_features(_FakeEngine(), config, "paris")


def test_create_pg_engine_handles_password_special_characters():
    config = PostgresFeatureConfig("localhost", 5432, "velib_dw", "velib", "p@:/word")

    engine = create_pg_engine(config)

    assert engine.url.render_as_string(hide_password=False) == "postgresql+psycopg2://velib:p%40%3A%2Fword@localhost:5432/velib_dw"


@pytest.mark.parametrize("predict_bikes", [True, False])
def test_load_training_actuals_for_dt_reads_side_specific_columns(monkeypatch, predict_bikes):
    captured = {}
    target_spec = target_spec_from_predict_bikes(predict_bikes)

    def fake_read_sql_query(sql, connection, params):
        captured["sql"] = str(sql)
        return pd.DataFrame(
            [
                {
                    "station_id": "station-001",
                    target_spec.paired_target_column: 10,
                    target_spec.label_column: 0,
                }
            ]
        )

    monkeypatch.setattr(pd, "read_sql_query", fake_read_sql_query)

    config = PostgresFeatureConfig("localhost", 5432, "velib_dw", "velib", "velib")
    df = load_training_actuals_for_dt(
        _FakeEngine(),
        config,
        "paris",
        "2026-03-11-10-05",
        predict_bikes=predict_bikes,
    )

    assert target_spec.paired_target_column in df.columns
    assert target_spec.label_column in df.columns
    assert f'"{target_spec.paired_target_column}" AS "{target_spec.paired_target_column}"' in captured["sql"]
