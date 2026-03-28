import pandas as pd
from src.features import postgres_store


class _FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeEngine:
    def connect(self):
        return _FakeConnection()


def _feature_config() -> postgres_store.PostgresFeatureConfig:
    return postgres_store.PostgresFeatureConfig(
        pg_host="dw-postgres",
        pg_port=5432,
        pg_db="velib_dw",
        pg_user="velib",
        pg_password="velib",
    )


def test_load_latest_serving_features_drops_excessively_stale_rows(monkeypatch):
    sample = pd.DataFrame(
        [
            {
                "city": "paris",
                "dt": "2026-03-27-18-10",
                "station_id": "stale-station",
                "capacity": 10,
                "lat": 1.0,
                "lon": 2.0,
                "bikes": 3,
                "docks": 7,
                "minutes_since_prev_snapshot": 5.0,
                "util_bikes": 0.3,
                "util_docks": 0.7,
                "delta_bikes_5m": 0.0,
                "delta_docks_5m": 0.0,
                "roll15_net_bikes": 0.0,
                "roll30_net_bikes": 0.0,
                "roll60_net_bikes": 0.0,
                "roll15_bikes_mean": 3.0,
                "roll30_bikes_mean": 3.0,
                "roll60_bikes_mean": 3.0,
                "nbr_bikes_weighted": pd.NA,
                "nbr_docks_weighted": pd.NA,
                "has_neighbors_within_radius": 0,
                "neighbor_count_within_radius": 0,
                "hour": 18,
                "dow": 5,
                "is_weekend": 0,
                "is_holiday": 0,
                "temperature_c": 10.0,
                "humidity_pct": 50.0,
                "wind_speed_ms": 1.0,
                "precipitation_mm": 0.0,
                "weather_code": 1.0,
                "hourly_temperature_c": 10.0,
                "hourly_humidity_pct": 50.0,
                "hourly_wind_speed_ms": 1.0,
                "hourly_precipitation_mm": 0.0,
                "hourly_precipitation_probability_pct": 0.0,
                "hourly_weather_code": 1.0,
            },
            {
                "city": "paris",
                "dt": "2026-03-28-04-40",
                "station_id": "fresh-station",
                "capacity": 10,
                "lat": 1.0,
                "lon": 2.0,
                "bikes": 4,
                "docks": 6,
                "minutes_since_prev_snapshot": 5.0,
                "util_bikes": 0.4,
                "util_docks": 0.6,
                "delta_bikes_5m": 0.0,
                "delta_docks_5m": 0.0,
                "roll15_net_bikes": 0.0,
                "roll30_net_bikes": 0.0,
                "roll60_net_bikes": 0.0,
                "roll15_bikes_mean": 4.0,
                "roll30_bikes_mean": 4.0,
                "roll60_bikes_mean": 4.0,
                "nbr_bikes_weighted": pd.NA,
                "nbr_docks_weighted": pd.NA,
                "has_neighbors_within_radius": 0,
                "neighbor_count_within_radius": 0,
                "hour": 4,
                "dow": 6,
                "is_weekend": 1,
                "is_holiday": 0,
                "temperature_c": 12.0,
                "humidity_pct": 60.0,
                "wind_speed_ms": 1.5,
                "precipitation_mm": 0.0,
                "weather_code": 1.0,
                "hourly_temperature_c": 12.0,
                "hourly_humidity_pct": 60.0,
                "hourly_wind_speed_ms": 1.5,
                "hourly_precipitation_mm": 0.0,
                "hourly_precipitation_probability_pct": 0.0,
                "hourly_weather_code": 1.0,
            },
        ]
    )

    monkeypatch.setattr(pd, "read_sql_query", lambda *args, **kwargs: sample.copy())

    result = postgres_store.load_latest_serving_features(
        _FakeEngine(),
        _feature_config(),
        "paris",
        max_dt_skew_minutes=60,
    )

    assert result["station_id"].tolist() == ["fresh-station"]


def test_load_latest_serving_features_logs_when_weather_context_is_sparse(monkeypatch, capsys):
    base_row = {
        "city": "paris",
        "dt": "2026-03-28-04-40",
        "capacity": 10,
        "lat": 1.0,
        "lon": 2.0,
        "bikes": 4,
        "docks": 6,
        "minutes_since_prev_snapshot": 5.0,
        "util_bikes": 0.4,
        "util_docks": 0.6,
        "delta_bikes_5m": 0.0,
        "delta_docks_5m": 0.0,
        "roll15_net_bikes": 0.0,
        "roll30_net_bikes": 0.0,
        "roll60_net_bikes": 0.0,
        "roll15_bikes_mean": 4.0,
        "roll30_bikes_mean": 4.0,
        "roll60_bikes_mean": 4.0,
        "nbr_bikes_weighted": pd.NA,
        "nbr_docks_weighted": pd.NA,
        "has_neighbors_within_radius": 0,
        "neighbor_count_within_radius": 0,
        "hour": 4,
        "dow": 6,
        "is_weekend": 1,
        "is_holiday": 0,
        "temperature_c": pd.NA,
        "humidity_pct": pd.NA,
        "wind_speed_ms": pd.NA,
        "precipitation_mm": pd.NA,
        "weather_code": pd.NA,
        "hourly_temperature_c": pd.NA,
        "hourly_humidity_pct": pd.NA,
        "hourly_wind_speed_ms": pd.NA,
        "hourly_precipitation_mm": pd.NA,
        "hourly_precipitation_probability_pct": pd.NA,
        "hourly_weather_code": pd.NA,
    }
    sample = pd.DataFrame(
        [
            {**base_row, "station_id": "station-1"},
            {**base_row, "station_id": "station-2"},
            {
                **base_row,
                "station_id": "station-3",
                "temperature_c": 12.0,
                "humidity_pct": 60.0,
                "wind_speed_ms": 1.5,
                "precipitation_mm": 0.0,
                "weather_code": 1.0,
                "hourly_temperature_c": 12.0,
                "hourly_humidity_pct": 60.0,
                "hourly_wind_speed_ms": 1.5,
                "hourly_precipitation_mm": 0.0,
                "hourly_precipitation_probability_pct": 0.0,
                "hourly_weather_code": 1.0,
            },
        ]
    )

    monkeypatch.setattr(pd, "read_sql_query", lambda *args, **kwargs: sample.copy())

    result = postgres_store.load_latest_serving_features(
        _FakeEngine(),
        _feature_config(),
        "paris",
    )

    assert result["station_id"].tolist() == ["station-1", "station-2", "station-3"]
    captured = capsys.readouterr()
    assert "serving weather context sparse" in captured.out
