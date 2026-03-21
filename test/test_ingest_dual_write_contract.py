import importlib
import sys
from datetime import datetime, timezone


def _reload_module(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_weather_cli_defaults_to_dual_write_from_env(monkeypatch):
    monkeypatch.setenv("CITY", "paris")
    monkeypatch.setenv("BUCKET", "weather-bucket")
    monkeypatch.setenv("OPENWEATHER_API_KEY", "test-key")
    monkeypatch.setenv("PGHOST", "localhost")
    monkeypatch.setenv("PGPORT", "15432")
    monkeypatch.setenv("PGDATABASE", "velib_dw")
    monkeypatch.setenv("PGUSER", "velib")
    monkeypatch.setenv("PGPASSWORD", "velib")

    weather_ingest = _reload_module("src.ingest.weather_ingest")

    args = weather_ingest._build_arg_parser().parse_args([])

    assert args.city == "paris"
    assert args.raw_bucket == "weather-bucket"
    assert args.api_key == "test-key"
    assert args.conn_uri == "postgresql+psycopg2://velib:velib@localhost:15432/velib_dw"


def test_weather_import_is_lazy_about_s3_client(monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", " definitely-missing-profile ")

    weather_ingest = _reload_module("src.ingest.weather_ingest")

    assert weather_ingest._default_s3_client.cache_info().currsize == 0


def test_weather_hourly_dataframe_defaults_missing_precipitation_to_zero():
    weather_ingest = _reload_module("src.ingest.weather_ingest")
    observed_at = datetime(2026, 3, 21, 10, 0, tzinfo=timezone.utc)
    payload = {
        "current": {"dt": int(observed_at.timestamp())},
        "hourly": [
            {
                "dt": int(observed_at.replace(minute=30).timestamp()),
                "temp": 12.5,
                "humidity": 72,
                "wind_speed": 4.0,
                "pop": 0.25,
                "weather": [{"id": 500, "main": "Rain", "description": "light rain"}],
            }
        ],
        "_meta_ingest": {
            "snapshot_bucket_at_utc": observed_at.isoformat(),
            "source": "openweather-onecall-3.0",
        },
    }

    df = weather_ingest.weather_hourly_dataframe(payload, city="paris", run_id="run-1")

    assert df.loc[0, "precipitation_mm"] == 0.0


def test_holidays_cli_defaults_to_dual_write_from_env(monkeypatch):
    monkeypatch.setenv("BUCKET", "holidays-bucket")
    monkeypatch.setenv("PGHOST", "localhost")
    monkeypatch.setenv("PGPORT", "15432")
    monkeypatch.setenv("PGDATABASE", "velib_dw")
    monkeypatch.setenv("PGUSER", "velib")
    monkeypatch.setenv("PGPASSWORD", "velib")

    holidays_ingest = _reload_module("src.ingest.holidays_ingest")

    args = holidays_ingest._build_arg_parser().parse_args(["--year", "2026"])

    assert args.raw_bucket == "holidays-bucket"
    assert args.conn_uri == "postgresql+psycopg2://velib:velib@localhost:15432/velib_dw"


def test_holidays_import_is_lazy_about_s3_client(monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", " definitely-missing-profile ")

    holidays_ingest = _reload_module("src.ingest.holidays_ingest")

    assert holidays_ingest._default_s3_client.cache_info().currsize == 0
