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
    ec2_tf = Path("infra/terraform/modules/platform/iam_ec2.tf").read_text(encoding="utf-8")

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
    assert "aws_cloudwatch_event_rule" not in lambda_tf
    assert "events.amazonaws.com" not in lambda_tf
    assert "sagemaker:InvokeEndpoint" in ec2_tf
    assert "sagemaker:DescribeEndpoint" in ec2_tf
    assert "AmazonSageMakerFullAccess" not in ec2_tf
    assert 'values(var.sagemaker_endpoints)' in ec2_tf


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


def test_formal_repo_cleanup_removes_legacy_runtime_and_scheduler_surfaces():
    removed_paths = [
        ".github/workflows/predictor.yml",
        ".github/workflows/quality.yml",
        ".github/workflows/publish_metrics.yml",
        ".github/workflows/groundtruth-cron.yml",
        ".github/workflows/cd_staging.yml",
        ".github/workflows/promote_prod.yml",
        "src/features/update_partitions.py",
        "src/monitoring/build_baseline_from_capture.py",
        "src/monitoring/metrics/lambda_publish_psi.py",
        "docs/demo_checklist.md",
    ]
    for path in removed_paths:
        assert not Path(path).exists(), f"legacy path should be removed: {path}"

    schedules_dir = Path("src/monitoring/schedules")
    assert not schedules_dir.exists() or not any(schedules_dir.glob("*.py"))


def test_formal_docs_and_dags_reflect_single_ec2_airflow_runtime_path():
    architecture = Path("docs/architecture.md").read_text(encoding="utf-8")
    cicd = Path("docs/cicd.md").read_text(encoding="utf-8")
    deployment_guide = Path("docs/deployment_guide.md").read_text(encoding="utf-8")
    runbook = Path("docs/operations_runbook.md").read_text(encoding="utf-8")
    monitoring = Path("docs/operations_runbook.md").read_text(encoding="utf-8")
    operator_manual = Path("docs/plan_detail/current_state_to_enterprise_operator_manual.md").read_text(encoding="utf-8")
    production_dag_path = Path("airflow/dags/production_serving_dags.py")
    staging_dag_path = Path("airflow/dags/staging_serving_dags.py")

    for document in (architecture, cicd, deployment_guide, runbook, monitoring, operator_manual):
        assert "RAW_S3_BUCKET" not in document
        assert "WEATHER_CITY" not in document
        assert "DW_HOST" not in document
        assert "promote_prod.yml" not in document
        assert "cd_staging.yml" not in document

    production_dag_source = production_dag_path.read_text(encoding="utf-8")
    staging_dag_source = staging_dag_path.read_text(encoding="utf-8")
    compile(production_dag_source, str(production_dag_path), "exec")
    compile(staging_dag_source, str(staging_dag_path), "exec")

    assert "serving_prediction_15min" in production_dag_source
    assert "serving_quality_backfill_15min" in production_dag_source
    assert "serving_metrics_publish_hourly" in production_dag_source
    assert "serving_psi_publish_hourly" in production_dag_source
    assert "staging_prediction_15min" in staging_dag_source
    assert "staging_quality_backfill_15min" in staging_dag_source
    assert "staging_metrics_publish_hourly" in staging_dag_source
    assert "staging_psi_publish_hourly" in staging_dag_source

    assert "staging_prediction_15min" in deployment_guide
    assert "staging_quality_backfill_15min" in deployment_guide
    assert "staging_metrics_publish_hourly" in deployment_guide
    assert "staging_psi_publish_hourly" in deployment_guide
    assert "serving_prediction_15min" in deployment_guide
    assert "staging_prediction_15min" in operator_manual
    assert "staging_quality_backfill_15min" in operator_manual


def test_compose_split_keeps_ec2_base_clean_and_local_override_explicit():
    base_compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    local_compose = Path("docker-compose.local.yml").read_text(encoding="utf-8")

    assert "AIRFLOW__CORE__EXECUTOR: CeleryExecutor" in base_compose
    assert "AIRFLOW__WEBSERVER__SECRET_KEY" in base_compose
    assert "redis:" in base_compose
    assert "airflow-worker-tier1:" in base_compose
    assert "airflow-worker-tier2:" in base_compose
    assert "exec airflow celery worker" in base_compose
    assert "--queues tier1" in base_compose
    assert "--queues tier2,default" in base_compose
    assert "airflow-worker-tier1:" in local_compose
    assert "airflow-worker-tier2:" in local_compose
    assert "./model_dir:/opt/airflow/model_dir" in base_compose
    assert "./model_dir:/app/model_dir:ro" in base_compose
    assert "AWS_PROFILE: Shirley-fr" not in base_compose
    assert "${USERPROFILE}/.aws/config" not in base_compose
    assert "${USERPROFILE}/.aws/credentials" not in base_compose
    assert "${USERPROFILE}/.aws/sso/cache" not in base_compose

    assert "AWS_PROFILE: Shirley-fr" in local_compose
    assert "${USERPROFILE}/.aws/config" in local_compose
    assert "${USERPROFILE}/.aws/credentials" in local_compose
    assert "${USERPROFILE}/.aws/sso/cache" in local_compose


def test_serving_dag_sensors_align_with_30_min_label_maturity():
    factory_source = Path("airflow/dags/serving_dag_factory.py").read_text(encoding="utf-8")

    assert "QUALITY_LABEL_MATURITY_MINUTES = 30" in factory_source
    assert "QUALITY_START_LAG_MINUTES = 7" in factory_source
    assert "QUALITY_TO_PREDICTION_DELTA = timedelta(minutes=QUALITY_LABEL_MATURITY_MINUTES + QUALITY_START_LAG_MINUTES)" in factory_source
    assert "METRICS_TO_QUALITY_DELTA = timedelta(minutes=5)" in factory_source
    assert "execution_delta=QUALITY_TO_PREDICTION_DELTA" in factory_source
    assert "execution_delta=METRICS_TO_QUALITY_DELTA" in factory_source
    assert 'external_task_id=f"predict_{target}"' in factory_source
    assert 'external_task_id=f"backfill_quality_{target}"' in factory_source
    assert "wait_for_metrics_dag" not in factory_source
    assert "queue=_tier1_queue()" in factory_source
    assert "queue=_tier2_queue()" in factory_source
    assert "pool=_serving_prediction_pool()" in factory_source
    assert "pool=_serving_observability_pool()" in factory_source


def test_feature_future_window_label_smoke_test_is_runtime_scoped():
    test_sql = Path("dbt/bikeshare_dbt/tests/feat_station_snapshot_5min_future_window_labels.sql").read_text(
        encoding="utf-8"
    )

    assert "from {{ ref('int_station_status_enriched') }}" in test_sql
    assert "runtime_window_start_utc_expr(default_lookback_hours=72)" in test_sql
    assert "{{ runtime_utc_expr('test_window_end_utc') }} + interval '30 minutes'" in test_sql
    assert "{{ runtime_utc_expr('test_window_end_utc') }}" in test_sql
    assert "from mature_feature_rows cur" in test_sql


def test_feature_label_maturity_consistency_test_is_runtime_scoped():
    test_sql = Path(
        "dbt/bikeshare_dbt/tests/feat_station_snapshot_5min_label_maturity_consistency.sql"
    ).read_text(encoding="utf-8")

    assert "runtime_window_start_utc_expr(default_lookback_hours=72)" in test_sql
    assert "{{ runtime_utc_expr('test_window_end_utc') }}" in test_sql
    assert "snapshot_bucket_at_utc + interval '30 minutes' <= {{ runtime_utc_expr('test_window_end_utc') }}" in test_sql


def test_feature_latest_window_null_test_uses_immature_window_not_full_horizon():
    test_sql = Path("dbt/bikeshare_dbt/tests/feat_station_snapshot_5min_latest_window_labels_null.sql").read_text(
        encoding="utf-8"
    )

    assert "feature_snapshot_step_minutes" in test_sql
    assert "feature_label_horizon_minutes" in test_sql
    assert "immature_window_minutes = label_horizon_minutes" in test_sql
    assert "interval '{{ immature_window_minutes }} minutes'" in test_sql


def test_tiered_dbt_dags_use_explicit_queue_and_pool_assignments():
    hotpath = Path("airflow/dags/dbt_station_status_hotpath_dag.py").read_text(encoding="utf-8")
    feature = Path("airflow/dags/dbt_feature_build_dag.py").read_text(encoding="utf-8")
    quality = Path("airflow/dags/dbt_quality_hourly_dag.py").read_text(encoding="utf-8")
    weather = Path("airflow/dags/weather_ingestion_dag.py").read_text(encoding="utf-8")

    assert 'return _get_setting("DBT_HOTPATH_POOL", "DBT_HOTPATH_POOL", "dbt_hotpath_pool")' in hotpath
    assert 'return _get_setting("AIRFLOW_TIER1_QUEUE", "AIRFLOW_TIER1_QUEUE", "tier1")' in hotpath
    assert 'return _get_setting("DBT_FEATURE_POOL", "DBT_FEATURE_POOL", "dbt_feature_pool")' in feature
    assert 'return _get_setting("AIRFLOW_TIER1_QUEUE", "AIRFLOW_TIER1_QUEUE", "tier1")' in feature
    assert 'return _get_setting("DBT_QUALITY_POOL", "DBT_QUALITY_POOL", "dbt_quality_pool")' in quality
    assert 'return _get_setting("AIRFLOW_TIER2_QUEUE", "AIRFLOW_TIER2_QUEUE", "tier2")' in quality
    assert 'return _get_setting("AIRFLOW_TIER1_QUEUE", "AIRFLOW_TIER1_QUEUE", "tier1")' in weather
    assert "max_active_runs=1" in weather


def test_dbt_runtime_selectors_and_thread_defaults_are_hotpath_safe():
    selectors = Path("dbt/bikeshare_dbt/selectors.yml").read_text(encoding="utf-8")
    feature = Path("airflow/dags/dbt_feature_build_dag.py").read_text(encoding="utf-8")
    hotpath = Path("airflow/dags/dbt_station_status_hotpath_dag.py").read_text(encoding="utf-8")
    quality = Path("airflow/dags/dbt_quality_hourly_dag.py").read_text(encoding="utf-8")
    diagnostic = Path("airflow/dags/dbt_diagnostic_daily_dag.py").read_text(encoding="utf-8")

    for selector_name in (
        "hf_feature_smoke_tests",
        "hf_station_status_smoke_tests",
        "hourly_quality_gate_tests",
        "daily_deep_quality_tests",
    ):
        assert f"- name: {selector_name}" in selectors

    assert '"DBT_FEATURE_REBUILD_LOOKBACK_MINUTES", "DBT_FEATURE_REBUILD_LOOKBACK_MINUTES", "120"' in feature
    assert "DBT_FEATURE_BUILD_TEST_SELECTOR" in feature
    assert "DBT_STATION_STATUS_HOTPATH_TEST_SELECTOR" in hotpath
    assert "DBT_QUALITY_TEST_SELECTOR" in quality
    assert "DBT_DEEP_QUALITY_TEST_SELECTOR" in diagnostic

    assert '"DBT_THREADS", "DBT_THREADS", "2"' in feature
    assert '"DBT_THREADS", "DBT_THREADS", "2"' in hotpath
    assert '"DBT_THREADS", "DBT_THREADS", "2"' in quality
    assert '"DBT_THREADS", "DBT_THREADS", "2"' in diagnostic


def test_hotpath_tests_are_retiered_out_of_quality_gate():
    hotpath_unique = Path("dbt/bikeshare_dbt/tests/fct_station_status_unique_grain.sql").read_text(encoding="utf-8")
    enriched_unique = Path("dbt/bikeshare_dbt/tests/int_station_status_enriched_unique_grain.sql").read_text(
        encoding="utf-8"
    )
    weather_coverage = Path("dbt/bikeshare_dbt/tests/int_station_status_enriched_weather_context_coverage.sql").read_text(
        encoding="utf-8"
    )

    assert "hf_hotpath_smoke" in hotpath_unique
    assert "quality_gate" not in hotpath_unique
    assert "hf_hotpath_smoke" in enriched_unique
    assert "quality_gate" not in enriched_unique
    assert "quality_gate" in weather_coverage
