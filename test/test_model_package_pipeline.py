from pathlib import Path
import tarfile

from pipelines import deploy_via_sagemaker_sdk, export_and_upload_model, promote, rollback
from src.model_package import (
    activate_package,
    ensure_package_dir,
    load_deployment_state,
    load_package_manifest,
    resolve_active_package_dir,
    write_package_manifest,
)


def _write_package(tmp_path) -> Path:
    package_dir = ensure_package_dir("paris_model", "run-123", root_dir=tmp_path / "packages")
    (package_dir / "model" / "MLmodel").write_text("artifact_path: model\n", encoding="utf-8")
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
    (package_dir / "artifacts" / "eval_summary.json").write_text("{}", encoding="utf-8")
    return package_dir


def test_package_manifest_round_trip_and_activation(tmp_path):
    package_dir = _write_package(tmp_path)
    deployment_state_path = activate_package(package_dir, tmp_path / "deployments" / "bikes" / "local.json", source="pytest")

    manifest = load_package_manifest(package_dir)
    deployment_state = load_deployment_state(deployment_state_path)

    assert manifest["model_name"] == "paris_model"
    assert deployment_state["target_name"] == "bikes"
    assert deployment_state["package_dir"] == str(package_dir.resolve())
    assert resolve_active_package_dir(deployment_state_path=deployment_state_path) == package_dir.resolve()


def test_export_package_creates_tar_from_local_package(tmp_path):
    package_dir = _write_package(tmp_path)

    result = export_and_upload_model.main(
        ["--package-dir", str(package_dir), "--output-dir", str(tmp_path / "dist")]
    )

    assert Path(result["tar_path"]).exists()
    assert result["model_name"] == "paris_model"
    with tarfile.open(result["tar_path"], "r:gz") as tar:
        names = tar.getnames()
    assert "MLmodel" in names
    assert "model/MLmodel" not in names


def test_deploy_wrapper_writes_environment_deployment_state(monkeypatch, tmp_path):
    package_dir = _write_package(tmp_path)
    calls = []

    class _FakeWaiter:
        def wait(self, EndpointName):
            calls.append(("wait", EndpointName))

    class _FakeSageMaker:
        def create_model(self, **kwargs):
            calls.append(("create_model", kwargs["ModelName"]))

        def create_endpoint_config(self, **kwargs):
            calls.append(("create_endpoint_config", kwargs["EndpointConfigName"]))

        def describe_endpoint(self, EndpointName):
            raise deploy_via_sagemaker_sdk.ClientError(
                {"Error": {"Code": "ValidationException", "Message": "Could not find endpoint"}},
                "DescribeEndpoint",
            )

        def create_endpoint(self, **kwargs):
            calls.append(("create_endpoint", kwargs["EndpointName"]))

        def update_endpoint(self, **kwargs):
            calls.append(("update_endpoint", kwargs["EndpointName"]))

        def get_waiter(self, name):
            return _FakeWaiter()

    monkeypatch.setattr(deploy_via_sagemaker_sdk.boto3, "client", lambda service_name, region_name=None: _FakeSageMaker())

    result = deploy_via_sagemaker_sdk.main(
        [
            "--endpoint-name",
            "bikeshare-bikes-staging",
            "--role-arn",
            "arn:aws:iam::123456789012:role/sm-exec",
            "--image-uri",
            "123456789012.dkr.ecr.eu-west-3.amazonaws.com/mlflow:latest",
            "--package-s3-uri",
            "s3://bucket/packages/model.tar.gz",
            "--package-dir",
            str(package_dir),
            "--instance-type",
            "ml.m5.large",
            "--region",
            "eu-west-3",
            "--environment",
            "staging",
            "--deployment-state-path",
            str(tmp_path / "deployments" / "bikes" / "staging.json"),
        ]
    )

    deployment_state = load_deployment_state(result["deployment_state_path"])
    assert deployment_state["environment"] == "staging"
    assert deployment_state["endpoint_name"] == "bikeshare-bikes-staging"
    assert any(item[0] == "create_model" for item in calls)


def test_deploy_cleans_up_new_resources_on_failure(monkeypatch, tmp_path):
    package_dir = _write_package(tmp_path)
    calls = []

    class _FakeSageMaker:
        def create_model(self, **kwargs):
            calls.append(("create_model", kwargs["ModelName"]))

        def create_endpoint_config(self, **kwargs):
            calls.append(("create_endpoint_config", kwargs["EndpointConfigName"]))

        def describe_endpoint(self, EndpointName):
            raise deploy_via_sagemaker_sdk.ClientError(
                {"Error": {"Code": "ValidationException", "Message": "Could not find endpoint"}},
                "DescribeEndpoint",
            )

        def create_endpoint(self, **kwargs):
            calls.append(("create_endpoint", kwargs["EndpointName"]))
            raise RuntimeError("create endpoint failed")

        def delete_endpoint_config(self, **kwargs):
            calls.append(("delete_endpoint_config", kwargs["EndpointConfigName"]))

        def delete_model(self, **kwargs):
            calls.append(("delete_model", kwargs["ModelName"]))

    monkeypatch.setattr(deploy_via_sagemaker_sdk.boto3, "client", lambda service_name, region_name=None: _FakeSageMaker())

    try:
        deploy_via_sagemaker_sdk.main(
            [
                "--endpoint-name",
                "bikeshare-bikes-staging",
                "--role-arn",
                "arn:aws:iam::123456789012:role/sm-exec",
                "--image-uri",
                "123456789012.dkr.ecr.eu-west-3.amazonaws.com/mlflow:latest",
                "--package-s3-uri",
                "s3://bucket/packages/model.tar.gz",
                "--package-dir",
                str(package_dir),
                "--instance-type",
                "ml.m5.large",
                "--region",
                "eu-west-3",
                "--environment",
                "staging",
                "--deployment-state-path",
                str(tmp_path / "deployments" / "bikes" / "staging.json"),
            ]
        )
    except RuntimeError as exc:
        assert str(exc) == "create endpoint failed"
    else:
        raise AssertionError("expected deployment failure")

    assert any(item[0] == "delete_endpoint_config" for item in calls)
    assert any(item[0] == "delete_model" for item in calls)
    assert not (tmp_path / "deployments" / "bikes" / "staging.json").exists()


def test_promote_copies_deployment_state_to_target_environment(tmp_path):
    package_dir = _write_package(tmp_path)
    source_state_path = activate_package(package_dir, tmp_path / "deployments" / "bikes" / "staging.json", source="pytest")

    result = promote.main(
        [
            "--source-deployment-state-path",
            str(source_state_path),
            "--target-deployment-state-path",
            str(tmp_path / "deployments" / "bikes" / "production.json"),
            "--target-environment",
            "production",
        ]
    )

    promoted_state = load_deployment_state(result["target_deployment_state_path"])
    assert promoted_state["environment"] == "production"
    assert promoted_state["package_dir"] == str(package_dir.resolve())


def test_rollback_restores_previous_deployment_state(tmp_path):
    package_dir = _write_package(tmp_path)
    current_state_path = Path(
        activate_package(package_dir, tmp_path / "deployments" / "bikes" / "production.json", source="pytest")
    )

    previous_package_dir = ensure_package_dir("paris_model_prev", "run-122", root_dir=tmp_path / "packages")
    (previous_package_dir / "model" / "MLmodel").write_text("artifact_path: model\n", encoding="utf-8")
    previous_manifest = {
        **load_package_manifest(package_dir),
        "model_name": "paris_model_prev",
        "run_id": "run-122",
        "paths": {
            "package_dir": str(previous_package_dir.resolve()),
            "model_dir": str((previous_package_dir / "model").resolve()),
            "package_manifest_path": str((previous_package_dir / "package_manifest.json").resolve()),
            "artifacts_dir": str((previous_package_dir / "artifacts").resolve()),
        },
    }
    write_package_manifest(previous_package_dir, previous_manifest)
    previous_state_path = Path(
        activate_package(previous_package_dir, tmp_path / "deployments" / "bikes" / "previous_prod.json", source="pytest")
    )

    result = rollback.main(
        [
            "--target-name",
            "bikes",
            "--environment",
            "production",
            "--from-state",
            str(current_state_path),
            "--to-state",
            str(previous_state_path),
        ]
    )

    restored_state = load_deployment_state(result["restored_deployment_state_path"])
    assert restored_state["source"] == "rollback"
    assert restored_state["environment"] == "production"
    assert restored_state["package_dir"] == str(previous_package_dir.resolve())
