import pytest

pytest.importorskip("mlflow")

from src.model_package import compute_package_sha256, ensure_package_dir, load_package_manifest, write_package_manifest
from src.model_target import target_spec_from_predict_bikes
from src.orchestration import retrain


def _base_config(tmp_path, reason: str, predict_bikes: bool = True) -> retrain.RetrainConfig:
    return retrain.RetrainConfig(
        reason=reason,
        city="paris",
        predict_bikes=predict_bikes,
        model_type="xgboost",
        lookback_days=30,
        pg_host="localhost",
        pg_port=15432,
        pg_db="velib_dw",
        pg_user="velib",
        pg_password="velib",
        pg_schema="analytics",
        feature_table="feat_station_snapshot_5min",
        experiment="pytest-retrain",
        dbt_project_dir="dbt/bikeshare_dbt",
        dbt_profiles_dir="dbt",
        max_feature_age_hours=48,
        summary_path=str(tmp_path / f"{reason}_summary.json"),
    )


@pytest.mark.parametrize("predict_bikes", [True, False])
def test_retrain_smoke_builds_candidate_summary(monkeypatch, tmp_path, predict_bikes):
    calls = {"dbt": 0, "register": 0, "package": 0, "tags": 0}
    target_spec = target_spec_from_predict_bikes(predict_bikes)

    monkeypatch.setattr(retrain, "check_feature_freshness", lambda config: None)
    monkeypatch.setattr(
        retrain,
        "get_latest_feature_dt",
        lambda config: retrain.datetime(2026, 3, 11, 0, 0, tzinfo=retrain.timezone.utc),
    )
    monkeypatch.setattr(retrain, "run_dbt_refresh", lambda config: calls.__setitem__("dbt", calls["dbt"] + 1))
    monkeypatch.setattr(
        retrain,
        "run_training_job",
        lambda config, latest_dt: {
            "run_id": f"run-{config.reason}",
            "model_name": f"paris_{target_spec.label_column}_xgboost",
            "model_artifact_path": "model",
            "package_dir": str(tmp_path / "packages" / "run"),
            "package_manifest_path": str(tmp_path / "packages" / "run" / "package_manifest.json"),
            "predict_bikes": predict_bikes,
            "target_name": target_spec.target_name,
            "label": target_spec.label_column,
            "label_column": target_spec.label_column,
            "best_threshold": 0.33,
            "pr_auc_valid": 0.71,
            "overfit_gap": 0.04,
            "feature_contract": "v1_dim_weather_aligned",
            "feature_source": "analytics.feat_station_snapshot_5min",
            "time_start": "2026-02-10 00:00",
            "time_end": "2026-03-11 00:00",
        },
    )

    def fake_register(summary, config, model_name):
        calls["register"] += 1
        return {
            "version": str(calls["register"]),
            "model_name": model_name,
            "best_threshold": summary["best_threshold"],
            "label": summary["label"],
        }

    monkeypatch.setattr(retrain, "register_candidate_model", fake_register)
    monkeypatch.setattr(
        retrain,
        "update_registered_package_metadata",
        lambda summary, registration, model_name: calls.__setitem__("package", calls["package"] + 1)
        or {"package_sha256": "sha"},
    )
    monkeypatch.setattr(
        retrain,
        "update_registration_package_tags",
        lambda registration, package_metadata: calls.__setitem__("tags", calls["tags"] + 1),
    )

    candidate_summary = retrain.run_retraining(_base_config(tmp_path, "manual", predict_bikes))

    assert candidate_summary["reason"] == "manual"
    assert candidate_summary["next_action"] == "deploy_candidate"
    assert candidate_summary["training"]["label"] == target_spec.label_column
    assert candidate_summary["registration"]["version"] == "1"
    assert candidate_summary["summary_path"].endswith("manual_summary.json")
    assert calls["dbt"] == 1
    assert calls["register"] == 1
    assert calls["package"] == 1
    assert calls["tags"] == 1


def test_run_retraining_refreshes_before_freshness_and_uses_post_refresh_latest_dt(monkeypatch, tmp_path):
    call_order = []

    monkeypatch.setattr(retrain, "run_dbt_refresh", lambda config: call_order.append("dbt_refresh"))
    monkeypatch.setattr(retrain, "check_feature_freshness", lambda config: call_order.append("freshness"))
    monkeypatch.setattr(
        retrain,
        "get_latest_feature_dt",
        lambda config: call_order.append("latest_dt")
        or retrain.datetime(2026, 3, 12, 0, 0, tzinfo=retrain.timezone.utc),
    )

    captured_latest_dt = {}

    def fake_run_training_job(config, latest_dt):
        call_order.append("training")
        captured_latest_dt["value"] = latest_dt
        return {
            "run_id": "run-manual",
            "model_name": "paris_y_stockout_bikes_30_xgboost",
            "model_artifact_path": "model",
            "package_dir": str(tmp_path / "packages" / "run"),
            "package_manifest_path": str(tmp_path / "packages" / "run" / "package_manifest.json"),
            "predict_bikes": True,
            "target_name": "bikes",
            "label": "y_stockout_bikes_30",
            "label_column": "y_stockout_bikes_30",
            "best_threshold": 0.33,
            "pr_auc_valid": 0.71,
            "overfit_gap": 0.04,
            "feature_contract": "v1_dim_weather_aligned",
            "feature_source": "analytics.feat_station_snapshot_5min",
            "time_start": "2026-02-10 00:00",
            "time_end": "2026-03-11 00:00",
        }

    monkeypatch.setattr(retrain, "run_training_job", fake_run_training_job)
    monkeypatch.setattr(retrain, "register_candidate_model", lambda summary, config, model_name: {"version": "1"})
    monkeypatch.setattr(retrain, "update_registered_package_metadata", lambda summary, registration, model_name: {})
    monkeypatch.setattr(retrain, "update_registration_package_tags", lambda registration, package_metadata: None)

    retrain.run_retraining(_base_config(tmp_path, "manual"))

    assert call_order == ["dbt_refresh", "freshness", "latest_dt", "training"]
    assert captured_latest_dt["value"] == retrain.datetime(2026, 3, 12, 0, 0, tzinfo=retrain.timezone.utc)


def test_run_training_job_uses_env_for_database_credentials(monkeypatch, tmp_path):
    config = _base_config(tmp_path, "manual", predict_bikes=False)
    config = retrain.RetrainConfig(**{**config.__dict__, "pg_password": "p@:/secret"})

    captured = {}

    class _Completed:
        def __init__(self):
            self.stdout = (
                'TRAINING_RESULT_JSON::{"run_id":"run-1","model_name":"paris_y_stockout_docks_30_xgboost",'
                '"model_artifact_path":"model","package_dir":"model_dir/packages/run-1",'
                '"package_manifest_path":"model_dir/packages/run-1/package_manifest.json",'
                '"predict_bikes":false,"target_name":"docks","label":"y_stockout_docks_30",'
                '"label_column":"y_stockout_docks_30","best_threshold":0.4,"pr_auc_valid":0.8,'
                '"overfit_gap":0.02,"feature_contract":"v1_dim_weather_aligned",'
                '"feature_source":"analytics.feat_station_snapshot_5min","time_start":"2026-02-10 00:00",'
                '"time_end":"2026-03-11 00:00"}'
            )
            self.stderr = ""

        def check_returncode(self):
            return None

    def fake_subprocess_run(command, cwd, check, capture_output, text, env):
        captured["command"] = command
        captured["env"] = env
        return _Completed()

    monkeypatch.setattr(retrain.subprocess, "run", fake_subprocess_run)

    latest_dt = retrain.datetime(2026, 3, 11, 0, 0, tzinfo=retrain.timezone.utc)
    result = retrain.run_training_job(config, latest_dt)

    assert "--pg-password" not in captured["command"]
    assert captured["env"]["PGPASSWORD"] == "p@:/secret"
    assert result["label"] == "y_stockout_docks_30"


def test_run_dbt_refresh_targets_configured_feature_table(monkeypatch, tmp_path):
    config = retrain.RetrainConfig(
        **{**_base_config(tmp_path, "manual").__dict__, "feature_table": "feat_station_snapshot_custom"}
    )
    calls = []

    monkeypatch.setattr(
        retrain,
        "run_dbt_command",
        lambda action, project_dir, profiles_dir, select_args, extra_args=None: calls.append((action, select_args)),
    )

    retrain.run_dbt_refresh(config)

    assert calls == [
        ("run", ["+feat_station_snapshot_custom"]),
        ("test", ["+feat_station_snapshot_custom"]),
    ]


def test_update_registered_package_metadata_hashes_finalized_manifest(tmp_path):
    package_dir = ensure_package_dir("paris_model", "run-123", root_dir=tmp_path / "packages")
    manifest = {
        "package_layout_version": "1",
        "created_at_utc": "2026-03-13T00:00:00Z",
        "model_name": "paris_model",
        "run_id": "run-123",
        "model_type": "xgboost",
        "predict_bikes": True,
        "target_name": "bikes",
        "label_column": "y_stockout_bikes_30",
        "paired_target_column": "target_bikes_t30",
        "score_column": "yhat_bikes",
        "score_bin_column": "yhat_bikes_bin",
        "actual_t30_column": "bikes_t30",
        "best_threshold": 0.33,
        "pr_auc_valid": 0.81,
        "overfit_gap": 0.04,
        "feature_contract_version": "v1_dim_weather_aligned",
        "feature_columns": ["minutes_since_prev_snapshot"],
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
    before = compute_package_sha256(package_dir)

    package_metadata = retrain.update_registered_package_metadata(
        {"package_dir": str(package_dir)},
        {"version": "7"},
        "registered-model",
    )

    final_manifest = load_package_manifest(package_dir)
    after = compute_package_sha256(package_dir)

    assert final_manifest["registered_model_name"] == "registered-model"
    assert final_manifest["registered_version"] == "7"
    assert final_manifest["aliases"] == ["candidate"]
    assert package_metadata["package_sha256"] == after
    assert before != after


def test_evaluate_candidate_rejects_low_quality():
    with pytest.raises(RuntimeError, match="candidate rejected: pr_auc_valid"):
        retrain.evaluate_candidate({"pr_auc_valid": 0.4, "overfit_gap": 0.01})


def test_check_feature_freshness_rejects_stale_features(monkeypatch, tmp_path):
    config = _base_config(tmp_path, "manual")
    monkeypatch.setattr(
        retrain,
        "get_latest_feature_dt",
        lambda cfg: retrain.datetime.now(retrain.timezone.utc) - retrain.timedelta(hours=72),
    )

    with pytest.raises(RuntimeError, match="feature table analytics.feat_station_snapshot_5min is stale"):
        retrain.check_feature_freshness(config)
