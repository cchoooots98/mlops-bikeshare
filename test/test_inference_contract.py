import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from src.features.schema import FEATURE_COLUMNS
from src.inference import featurize_online, predictor
from src.model_package import activate_package, ensure_package_dir, write_package_manifest
from src.model_target import target_spec_from_predict_bikes
from src.utils.config import RuntimeSettings

_SMOKE_SPEC = importlib.util.spec_from_file_location(
    "repo_smoke_invoke",
    Path(__file__).resolve().parent / "smoke_invoke.py",
)
smoke_invoke = importlib.util.module_from_spec(_SMOKE_SPEC)
assert _SMOKE_SPEC.loader is not None
_SMOKE_SPEC.loader.exec_module(smoke_invoke)


def _runtime_settings(tmp_path) -> RuntimeSettings:
    return RuntimeSettings(
        aws_region="eu-west-3",
        city="paris",
        bucket="unit-test-bucket",
        sm_endpoint="bikeshare-staging",
        pg_host="localhost",
        pg_port=5432,
        pg_db="velib_dw",
        pg_user="velib",
        pg_password="velib",
        pg_schema="analytics",
        training_feature_table="feat_station_snapshot_5min",
        online_feature_table="feat_station_snapshot_latest",
        model_package_dir=None,
        deployment_state_path=str(tmp_path / "deployments" / "local.json"),
        predict_bikes=True,
    )


def _online_features_df() -> pd.DataFrame:
    row = {"city": "paris", "dt": "2026-03-11-10-05", "station_id": "station-001"}
    for index, column in enumerate(FEATURE_COLUMNS):
        row[column] = float(index + 1)
    return pd.DataFrame([row])


def _write_package(tmp_path, *, feature_columns=None, predict_bikes=True) -> tuple[Path, Path]:
    target_spec = target_spec_from_predict_bikes(predict_bikes)
    package_dir = ensure_package_dir("paris_model", "run-123", root_dir=tmp_path / "packages")
    (package_dir / "model" / "MLmodel").write_text("artifact_path: model\n", encoding="utf-8")
    manifest = {
        "package_layout_version": "1",
        "created_at_utc": "2026-03-13T00:00:00Z",
        "model_name": "paris_model",
        "run_id": "run-123",
        "model_type": "xgboost",
        "predict_bikes": predict_bikes,
        "target_name": target_spec.target_name,
        "label_column": target_spec.label_column,
        "paired_target_column": target_spec.paired_target_column,
        "score_column": target_spec.score_column,
        "score_bin_column": target_spec.score_bin_column,
        "actual_t30_column": target_spec.actual_t30_column,
        "best_threshold": 0.37,
        "pr_auc_valid": 0.81,
        "overfit_gap": 0.04,
        "feature_contract_version": "v1_dim_weather_aligned",
        "feature_columns": list(feature_columns or FEATURE_COLUMNS),
        "feature_source": "analytics.feat_station_snapshot_5min",
        "city": "paris",
        "time_start": "2026-03-01 00:00",
        "time_end": "2026-03-07 23:55",
        "train_end_dt": "2026-03-06-23-55",
        "valid_start_dt": "2026-03-07-00-55",
        "registered_model_name": None,
        "registered_version": None,
        "aliases": [],
        "paths": {
            "package_dir": str(package_dir.resolve()),
            "model_dir": str((package_dir / "model").resolve()),
            "package_manifest_path": str((package_dir / "package_manifest.json").resolve()),
            "artifacts_dir": str((package_dir / "artifacts").resolve()),
        },
    }
    write_package_manifest(package_dir, manifest)
    deployment_state_path = Path(
        activate_package(package_dir, tmp_path / "deployments" / "local.json", source="pytest")
    )
    return package_dir, deployment_state_path


def test_smoke_invoke_payload_matches_feature_contract():
    payload = smoke_invoke.build_payload()

    assert payload["inputs"]["dataframe_split"]["columns"] == FEATURE_COLUMNS
    assert len(payload["inputs"]["dataframe_split"]["data"][0]) == len(FEATURE_COLUMNS)


@pytest.mark.parametrize(("predict_bikes", "expected_label"), [(True, "y_stockout_bikes_30"), (False, "y_stockout_docks_30")])
def test_load_prediction_manifest_requires_threshold_and_target(tmp_path, predict_bikes, expected_label):
    _write_package(tmp_path, predict_bikes=predict_bikes)

    metadata = predictor.load_prediction_manifest(deployment_state_path=tmp_path / "deployments" / "local.json")

    assert metadata["label"] == expected_label
    assert metadata["best_threshold"] == 0.37
    assert metadata["feature_columns"] == FEATURE_COLUMNS


def test_load_prediction_manifest_uses_package_feature_columns_without_runtime_constant_match(tmp_path):
    package_columns = ["util_bikes", "minutes_since_prev_snapshot"]
    _write_package(tmp_path, feature_columns=package_columns)

    metadata = predictor.load_prediction_manifest(deployment_state_path=tmp_path / "deployments" / "local.json")

    assert metadata["feature_columns"] == package_columns


def test_build_online_features_reads_latest_table_only(monkeypatch, tmp_path):
    settings = _runtime_settings(tmp_path)
    calls = {}
    selected_columns = ["util_bikes", "minutes_since_prev_snapshot"]

    monkeypatch.setattr(featurize_online, "load_runtime_settings", lambda: settings)
    monkeypatch.setattr(featurize_online, "create_pg_engine", lambda config: object())

    def fake_load_latest_serving_features(engine, config, city, select_columns):
        calls["online_table"] = config.online_table
        calls["city"] = city
        calls["select_columns"] = select_columns
        return _online_features_df()

    monkeypatch.setattr(featurize_online, "load_latest_serving_features", fake_load_latest_serving_features)

    df = featurize_online.build_online_features(feature_columns=selected_columns)

    assert calls["online_table"] == "feat_station_snapshot_latest"
    assert calls["city"] == "paris"
    assert calls["select_columns"] == ["city", "dt", "station_id", "capacity", "lat", "lon", "bikes", "docks", *selected_columns]
    assert list(df.columns) == ["city", "dt", "station_id", *selected_columns]


@pytest.mark.parametrize("predict_bikes", [True, False])
def test_predict_rowwise_uses_package_threshold(monkeypatch, predict_bikes):
    monkeypatch.setattr(predictor, "_invoke_endpoint_one", lambda endpoint, feature_row, inference_id, feature_columns: 0.2)
    features = _online_features_df()
    target_spec = target_spec_from_predict_bikes(predict_bikes)

    below = predictor._predict_rowwise_threaded(
        "endpoint",
        features,
        target_spec=target_spec,
        threshold=0.25,
        feature_columns=FEATURE_COLUMNS,
        max_workers=1,
    )
    above = predictor._predict_rowwise_threaded(
        "endpoint",
        features,
        target_spec=target_spec,
        threshold=0.15,
        feature_columns=FEATURE_COLUMNS,
        max_workers=1,
    )

    assert below.loc[0, "prediction_target"] == target_spec.target_name
    assert below.loc[0, target_spec.score_bin_column] == 0.0
    assert above.loc[0, target_spec.score_bin_column] == 1.0


def test_formal_codepaths_do_not_reference_legacy_feature_builder():
    forbidden = ("src.features.build_features", "features_offline", "athena_conn(", "read_env(", "engineer(")
    paths = [
        Path("src/inference/featurize_online.py"),
        Path("src/inference/predictor.py"),
        Path("src/monitoring/quality_backfill.py"),
    ]

    for path in paths:
        content = (Path(__file__).resolve().parents[1] / path).read_text(encoding="utf-8")
        for pattern in forbidden:
            assert pattern not in content, f"{path} still references legacy pattern: {pattern}"
