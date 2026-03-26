import json
from pathlib import Path

import pytest
from src.config import load_runtime_settings
from src.model_package import default_package_root_for_target
from src.training import train


def test_runtime_settings_reject_legacy_alias_only_configs(monkeypatch, tmp_path):
    runtime_defaults = {
        "Variables": {
            "AWS_REGION": "eu-west-3",
            "RAW_S3_BUCKET": "legacy-bucket",
            "WEATHER_CITY": "paris",
            "DW_HOST": "legacy-host",
            "DW_PORT": "15432",
            "DW_DB": "legacy-db",
            "DW_USER": "legacy-user",
            "DW_PASSWORD": "legacy-password",
        }
    }
    config_path = tmp_path / "env.json"
    config_path.write_text(json.dumps(runtime_defaults), encoding="utf-8")

    monkeypatch.setattr("src.config.runtime.DEFAULT_RUNTIME_CONFIG_PATH", config_path)
    monkeypatch.delenv("PGHOST", raising=False)
    monkeypatch.delenv("PGPORT", raising=False)
    monkeypatch.delenv("PGDATABASE", raising=False)
    monkeypatch.delenv("PGUSER", raising=False)
    monkeypatch.delenv("PGPASSWORD", raising=False)
    monkeypatch.delenv("CITY", raising=False)
    monkeypatch.delenv("BUCKET", raising=False)

    with pytest.raises(ValueError, match="missing required runtime settings"):
        load_runtime_settings()


@pytest.mark.parametrize(
    ("predict_bikes", "expected_root"),
    [
        (True, default_package_root_for_target("bikes")),
        (False, default_package_root_for_target("docks")),
    ],
)
def test_train_main_defaults_package_root_by_target(monkeypatch, predict_bikes, expected_root):
    captured = {}

    def fake_run_training_pipeline(data_config, train_config):
        captured["package_root"] = train_config.package_root
        return {"ok": True}

    monkeypatch.setattr(train, "run_training_pipeline", fake_run_training_pipeline)

    result = train.main(
        [
            "--city",
            "paris",
            "--start",
            "2026-03-01 00:00",
            "--end",
            "2026-03-01 01:00",
            "--pg-host",
            "localhost",
            "--pg-port",
            "15432",
            "--pg-db",
            "velib_dw",
            "--pg-user",
            "velib",
            "--pg-password",
            "velib",
            "--predict-bikes",
            str(predict_bikes).lower(),
        ]
    )

    assert result == {"ok": True}
    assert Path(captured["package_root"]) == expected_root


def test_runtime_settings_prefers_explicit_target_name_over_default_predict_bikes(monkeypatch, tmp_path):
    runtime_defaults = {
        "Variables": {
            "PGHOST": "localhost",
            "PGPORT": "15432",
            "PGDATABASE": "velib_dw",
            "PGUSER": "velib",
            "PGPASSWORD": "velib",
            "AWS_REGION": "eu-west-3",
            "CITY": "paris",
            "BUCKET": "bucket",
            "PREDICT_BIKES": "true",
            "TARGET_NAME": "bikes",
            "SERVING_ENVIRONMENT": "staging",
        }
    }
    config_path = tmp_path / "env.json"
    config_path.write_text(json.dumps(runtime_defaults), encoding="utf-8")

    monkeypatch.setattr("src.config.runtime.DEFAULT_RUNTIME_CONFIG_PATH", config_path)
    monkeypatch.setenv("TARGET_NAME", "docks")
    monkeypatch.delenv("PREDICT_BIKES", raising=False)

    settings = load_runtime_settings()

    assert settings.target_name == "docks"
    assert settings.predict_bikes is False
