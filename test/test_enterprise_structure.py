import importlib.util
import sys
from pathlib import Path

from src.serving import parse_router_request, resolve_endpoint_name


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_router_request_accepts_explicit_target_name():
    request = parse_router_request({"target_name": "docks", "environment": "staging"})

    assert request.target_name == "docks"
    assert request.predict_bikes is False
    assert request.environment == "staging"
    assert resolve_endpoint_name(target_name=request.target_name, environment=request.environment) == "bikeshare-docks-staging"


def test_dashboard_targeting_is_target_aware():
    targeting = _load_module(Path("app/dashboard/targeting.py"), "dashboard_targeting")
    config = targeting.resolve_dashboard_target(target_name="docks", city="paris", environment="production")

    assert config.target_name == "docks"
    assert config.label_column == "y_stockout_docks_30"
    assert config.score_column == "yhat_docks"
    assert config.endpoint_name == "bikeshare-docks-prod"


def test_formal_dashboard_entrypoint_excludes_legacy_debug_publish_controls():
    content = Path("app/dashboard.py").read_text(encoding="utf-8")

    assert "Publish Sample Metrics" not in content
    assert "features_offline" not in content
    assert "yhat_bikes" not in content


def test_formal_docs_use_target_specific_deployment_state_and_local_sqlite_mlflow():
    readme = Path("README.md").read_text(encoding="utf-8")
    architecture = Path("docs/architecture.md").read_text(encoding="utf-8")
    deployment_guide = Path("docs/deployment_guide.md").read_text(encoding="utf-8")
    

    assert "model_dir/deployments/local.json" not in readme
    assert "model_dir/deployments/local.json" not in architecture
    assert "http://localhost:5000" not in readme
    assert "http://localhost:5000" not in architecture
    assert "model_dir/deployments/bikes/local.json" in readme
    assert "sqlite:///model_dir/mlflow.db" in readme
    assert "--environment staging" in deployment_guide


def test_terraform_platform_module_has_no_placeholder_lambda():
    lambda_tf = Path("infra/terraform/modules/platform/lambda_eventbridge.tf").read_text(encoding="utf-8")
    cloudwatch_tf = Path("infra/terraform/modules/platform/cloudwatch.tf").read_text(encoding="utf-8")
    variables_tf = Path("infra/terraform/modules/platform/variables.tf").read_text(encoding="utf-8")

    assert "placeholder" not in lambda_tf.lower()
    assert "placeholder" not in cloudwatch_tf.lower()
    assert "sagemaker_endpoint_name" not in variables_tf
    assert "sagemaker_endpoints" in variables_tf
    assert "for_each" in cloudwatch_tf
    assert "aws_cloudwatch_dashboard" in cloudwatch_tf
    assert "PR-AUC-24h" in cloudwatch_tf
    assert "F1-24h" in cloudwatch_tf
    assert "PredictionHeartbeat" in cloudwatch_tf
    assert "PSI" in cloudwatch_tf


def test_terraform_uses_s3_native_locking_and_modern_version_floor():
    bootstrap_main = Path("infra/terraform/bootstrap/main.tf").read_text(encoding="utf-8")
    bootstrap_vars = Path("infra/terraform/bootstrap/variables.tf").read_text(encoding="utf-8")
    bootstrap_outputs = Path("infra/terraform/bootstrap/outputs.tf").read_text(encoding="utf-8")
    live_backend = Path("infra/terraform/live/backend.tf").read_text(encoding="utf-8")
    live_versions = Path("infra/terraform/live/versions.tf").read_text(encoding="utf-8")
    bootstrap_versions = Path("infra/terraform/bootstrap/versions.tf").read_text(encoding="utf-8")
    module_versions = Path("infra/terraform/modules/platform/versions.tf").read_text(encoding="utf-8")
    ci_workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    deployment_guide = Path("docs/deployment_guide.md").read_text(encoding="utf-8")
    operator_manual = Path("docs/plan_detail/current_state_to_enterprise_operator_manual.md").read_text(
        encoding="utf-8"
    )
    aws_runbook = Path("docs/plan_detail/day5_enterprise_aws_runbook.md").read_text(encoding="utf-8")
    cheatsheet = Path("docs/cheatsheet.md").read_text(encoding="utf-8")

    assert "aws_dynamodb_table" not in bootstrap_main
    assert "tf_lock_table_name" not in bootstrap_vars
    assert "tf_lock_table_name" not in bootstrap_outputs
    assert "use_lockfile = true" in live_backend
    assert 'required_version = ">= 1.10"' in live_versions
    assert 'required_version = ">= 1.10"' in bootstrap_versions
    assert 'required_version = ">= 1.10"' in module_versions
    assert "terraform_version: 1.13.3" in ci_workflow

    for doc in (deployment_guide, operator_manual, aws_runbook, cheatsheet):
        assert "dynamodb_table=" not in doc
        assert "TF_LOCK_TABLE" not in doc
        assert "tf_lock_table_name" not in doc
