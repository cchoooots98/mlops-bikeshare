import importlib.util
import re
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


def _read_optional_text(path: str) -> str | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    return file_path.read_text(encoding="utf-8")


def test_router_request_accepts_explicit_target_name():
    request = parse_router_request({"target_name": "docks", "environment": "staging"})

    assert request.target_name == "docks"
    assert request.predict_bikes is False
    assert request.environment == "staging"
    assert (
        resolve_endpoint_name(target_name=request.target_name, environment=request.environment)
        == "bikeshare-docks-staging"
    )


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


def test_ec2_release_update_guide_exists_and_covers_feature_reconcile():
    release_guide = Path("docs/ec2_release_update_guide.md").read_text(encoding="utf-8")
    deployment_guide = Path("docs/deployment_guide.md").read_text(encoding="utf-8")

    assert "DBT_FEATURE_REBUILD_LOOKBACK_MINUTES 10080" in release_guide
    assert "dbt_feature_build_5min" in release_guide
    assert "dbt_quality_hourly" in release_guide
    assert "ec2_release_update_guide.md" in deployment_guide


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
    assert "PSI_core" in cloudwatch_tf
    assert "PSI_weather" in cloudwatch_tf
    assert "aws_cloudwatch_event_rule" not in lambda_tf
    assert "events.amazonaws.com" not in lambda_tf
    assert "sagemaker:InvokeEndpoint" in ec2_tf
    assert "sagemaker:DescribeEndpoint" in ec2_tf
    assert "AmazonSageMakerFullAccess" not in ec2_tf
    assert "values(var.sagemaker_endpoints)" in ec2_tf


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
    operator_manual = _read_optional_text("docs/plan_detail/current_state_to_enterprise_operator_manual.md")
    aws_runbook = _read_optional_text("docs/plan_detail/day5_enterprise_aws_runbook.md")
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
        if doc is None:
            continue
        assert "dynamodb_table=" not in doc
        assert "TF_LOCK_TABLE" not in doc
        assert "tf_lock_table_name" not in doc


def test_terraform_oidc_subject_is_decoupled_from_project_resource_prefix():
    live_vars = Path("infra/terraform/live/variables.tf").read_text(encoding="utf-8")
    live_main = Path("infra/terraform/live/main.tf").read_text(encoding="utf-8")
    module_vars = Path("infra/terraform/modules/platform/variables.tf").read_text(encoding="utf-8")
    oidc_role = Path("infra/terraform/modules/platform/oidc_role.tf").read_text(encoding="utf-8")

    assert 'variable "github_repo_name"' in live_vars
    assert 'default = "mlops-bikeshare"' in live_vars
    assert "github_repo_name     = var.github_repo_name" in live_main

    assert 'variable "github_repo_name"' in module_vars
    assert "coalesce(var.github_repo_name, var.repo_name)" in module_vars
    assert 'data_bucket_name     = "${var.repo_name}-${local.account_id}-${var.aws_region}"' in module_vars

    assert "repo:${var.github_owner}/${local.github_repo_subject_name}:ref:refs/heads/main" in oidc_role
    assert "repo:${var.github_owner}/${var.repo_name}:ref:refs/heads/main" not in oidc_role


def test_formal_repo_runtime_surfaces_are_minimal_and_current():
    workflow_names = {path.name for path in Path(".github/workflows").glob("*.yml")}
    assert workflow_names == {"ci.yml"}

    metric_modules = {path.name for path in Path("src/monitoring/metrics").glob("*.py")}
    assert metric_modules == {
        "__init__.py",
        "metrics_helper.py",
        "publish_custom_metrics.py",
        "publish_psi.py",
        "publish_psi_all_targets.py",
    }

    schedules_dir = Path("src/monitoring/schedules")
    if schedules_dir.exists():
        assert {path.name for path in schedules_dir.iterdir()} <= {"__pycache__"}


def test_formal_docs_and_dags_reflect_single_ec2_airflow_runtime_path():
    architecture = Path("docs/architecture.md").read_text(encoding="utf-8")
    cicd = Path("docs/cicd.md").read_text(encoding="utf-8")
    deployment_guide = Path("docs/deployment_guide.md").read_text(encoding="utf-8")
    runbook = Path("docs/operations_runbook.md").read_text(encoding="utf-8")
    monitoring = Path("docs/operations_runbook.md").read_text(encoding="utf-8")
    operator_manual = _read_optional_text("docs/plan_detail/current_state_to_enterprise_operator_manual.md")
    production_dag_path = Path("airflow/dags/production_serving_dags.py")
    staging_dag_path = Path("airflow/dags/staging_serving_dags.py")

    for document in (architecture, cicd, deployment_guide, runbook, monitoring, operator_manual):
        if document is None:
            continue
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
    if operator_manual is not None:
        assert "staging_prediction_15min" in operator_manual
        assert "staging_quality_backfill_15min" in operator_manual


def test_compose_split_keeps_ec2_base_clean_and_local_override_explicit():
    base_compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    local_compose = Path("docker-compose.local.yml").read_text(encoding="utf-8")

    assert "AIRFLOW__CORE__EXECUTOR: CeleryExecutor" in base_compose
    assert "AIRFLOW__WEBSERVER__SECRET_KEY" in base_compose
    assert "redis:" in base_compose
    assert "exec airflow celery worker" in base_compose
    for worker_name, queue_name in (
        ("airflow-worker-core:", "core_5m"),
        ("airflow-worker-weather:", "weather_10m"),
        ("airflow-worker-serving:", "serving_rt"),
        ("airflow-worker-obs:", "obs_main"),
        ("airflow-worker-psi:", "obs_psi"),
        ("airflow-worker-sidecar:", "daily_sidecar"),
    ):
        assert worker_name in base_compose
        assert worker_name in local_compose
        assert f"--queues {queue_name}" in base_compose
    assert "airflow-worker-tier1:" not in base_compose
    assert "airflow-worker-tier2:" not in base_compose
    assert "airflow-worker-tier1:" not in local_compose
    assert "airflow-worker-tier2:" not in local_compose
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

    assert "from src.monitoring.quality_contract import QUALITY_LABEL_MATURITY_MINUTES" in factory_source
    assert "execution_date_fn_for_schedule(" in factory_source
    assert "minimum_age=timedelta(minutes=QUALITY_LABEL_MATURITY_MINUTES)" in factory_source
    assert "execution_date_fn=execution_date_fn_for_schedule(SERVING_QUALITY_SCHEDULE)" in factory_source
    assert 'external_task_id=f"predict_{target}"' in factory_source
    assert 'external_task_id=f"backfill_quality_{target}"' in factory_source
    assert "wait_for_metrics_dag" not in factory_source
    assert "queue=_serving_rt_queue()" in factory_source
    assert "queue=_obs_main_queue()" in factory_source
    assert "queue=_obs_psi_queue()" in factory_source
    assert "pool=_serving_prediction_pool()" in factory_source
    assert "pool=_serving_quality_metrics_pool()" in factory_source
    assert "pool=_serving_psi_pool()" in factory_source
    assert 'task_id="publish_psi_all_targets"' in factory_source
    assert '"src.monitoring.metrics.publish_psi_all_targets"' in factory_source
    assert "QUALITY_TO_PREDICTION_DELTA" not in factory_source
    assert "METRICS_TO_QUALITY_DELTA" not in factory_source
    assert "execution_delta=" not in factory_source
    assert "_serving_observability_pool" not in factory_source
    assert '"PSI_AGGREGATOR", "PSI_AGGREGATOR", "trimmed_mean"' in factory_source


def test_deployment_guide_serving_timing_contract_matches_schedule_defs():
    guide = Path("docs/deployment_guide.md").read_text(encoding="utf-8")
    schedules = Path("airflow/dags/schedule_defs.py").read_text(encoding="utf-8")

    assert 'SERVING_PREDICTION_SCHEDULE = "2-59/15 * * * *"' in schedules
    assert 'SERVING_QUALITY_SCHEDULE = "3-59/15 * * * *"' in schedules
    assert 'SERVING_METRICS_SCHEDULE = "42 * * * *"' in schedules
    assert 'SERVING_PSI_SCHEDULE = "12 * * * *"' in schedules

    assert "`staging_prediction_15min` / `serving_prediction_15min` run on `2,17,32,47 * * * *`" in guide
    assert "`staging_quality_backfill_15min` / `serving_quality_backfill_15min` run on `3,18,33,48 * * * *`" in guide
    assert "`staging_metrics_publish_hourly` / `serving_metrics_publish_hourly` run on `42 * * * *`" in guide
    assert "`staging_psi_publish_hourly` / `serving_psi_publish_hourly` run on `12 * * * *`" in guide
    assert "37 minutes earlier" not in guide
    assert "5 minutes earlier" not in guide


def test_airflow_dags_share_runtime_helpers_and_schedule_contracts():
    runtime_utils = Path("airflow/dags/runtime_utils.py").read_text(encoding="utf-8")
    schedule_defs = Path("airflow/dags/schedule_defs.py").read_text(encoding="utf-8")

    assert "def get_airflow_setting" in runtime_utils
    assert "def get_dw_connection" in runtime_utils
    assert "def get_dw_conn_uri" in runtime_utils

    for dag_path in (
        "airflow/dags/gbfs_ingestion_dag.py",
        "airflow/dags/weather_ingestion_dag.py",
        "airflow/dags/dbt_station_status_hotpath_dag.py",
        "airflow/dags/dbt_feature_build_dag.py",
        "airflow/dags/dbt_weather_refresh_dag.py",
        "airflow/dags/dbt_quality_hourly_dag.py",
        "airflow/dags/dbt_diagnostic_daily_dag.py",
        "airflow/dags/dbt_station_topology_daily_dag.py",
        "airflow/dags/holiday_yearly_dag.py",
        "airflow/dags/offline_retraining_dag.py",
        "airflow/dags/serving_dag_factory.py",
    ):
        content = Path(dag_path).read_text(encoding="utf-8")
        assert "def _get_setting(" not in content

    for dag_path in (
        "airflow/dags/gbfs_ingestion_dag.py",
        "airflow/dags/weather_ingestion_dag.py",
        "airflow/dags/holiday_yearly_dag.py",
    ):
        content = Path(dag_path).read_text(encoding="utf-8")
        assert "def _dw_conn_uri(" not in content
        assert re.search(r"(?<!get)_dw_conn_uri\(", content) is None
        assert "get_dw_conn_uri(" in content

    assert 'DBT_QUALITY_HOURLY_SCHEDULE = "13 * * * *"' in schedule_defs
    assert 'DBT_DIAGNOSTIC_DAILY_SCHEDULE = "47 2 * * *"' in schedule_defs
    assert 'HOLIDAY_YEARLY_SCHEDULE = "11 2 1 1 *"' in schedule_defs
    assert 'OFFLINE_MODEL_RETRAINING_DAILY_SCHEDULE = "30 3 * * *"' in schedule_defs

    assert "schedule=DBT_QUALITY_HOURLY_SCHEDULE" in Path("airflow/dags/dbt_quality_hourly_dag.py").read_text(
        encoding="utf-8"
    )
    assert "schedule=DBT_DIAGNOSTIC_DAILY_SCHEDULE" in Path("airflow/dags/dbt_diagnostic_daily_dag.py").read_text(
        encoding="utf-8"
    )
    assert "schedule=HOLIDAY_YEARLY_SCHEDULE" in Path("airflow/dags/holiday_yearly_dag.py").read_text(encoding="utf-8")
    assert "schedule=OFFLINE_MODEL_RETRAINING_DAILY_SCHEDULE" in Path(
        "airflow/dags/offline_retraining_dag.py"
    ).read_text(encoding="utf-8")

    gbfs = Path("airflow/dags/gbfs_ingestion_dag.py").read_text(encoding="utf-8")
    assert "def ingest_gbfs_feed_task" in gbfs
    assert "def ingest_station_information_task" not in gbfs
    assert "def ingest_station_status_task" not in gbfs


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
    test_sql = Path("dbt/bikeshare_dbt/tests/feat_station_snapshot_5min_label_maturity_consistency.sql").read_text(
        encoding="utf-8"
    )

    assert "runtime_window_start_utc_expr(default_lookback_hours=72)" in test_sql
    assert "{{ runtime_utc_expr('test_window_end_utc') }}" in test_sql
    assert "snapshot_bucket_at_utc + interval '30 minutes' <= f.latest_feature_snapshot_bucket_at_utc" in test_sql


def test_feature_targets_match_t30_test_is_runtime_scoped():
    test_sql = Path("dbt/bikeshare_dbt/tests/feat_station_snapshot_5min_targets_match_t30.sql").read_text(
        encoding="utf-8"
    )

    assert "runtime_window_start_utc_expr(default_lookback_hours=72)" in test_sql
    assert "{{ runtime_utc_expr('test_window_end_utc') }}" in test_sql
    assert "{{ runtime_utc_expr('test_window_end_utc') }} + interval '30 minutes'" in test_sql


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
    weather_refresh = Path("airflow/dags/dbt_weather_refresh_dag.py").read_text(encoding="utf-8")

    assert 'return _get_setting("DBT_HOTPATH_POOL", "DBT_HOTPATH_POOL", "dbt_hotpath_pool")' in hotpath
    assert 'return _get_setting("AIRFLOW_QUEUE_CORE_5M", "AIRFLOW_QUEUE_CORE_5M", CORE_5M_QUEUE)' in hotpath
    assert 'return _get_setting("DBT_FEATURE_POOL", "DBT_FEATURE_POOL", "dbt_feature_pool")' in feature
    assert 'return _get_setting("AIRFLOW_QUEUE_CORE_5M", "AIRFLOW_QUEUE_CORE_5M", CORE_5M_QUEUE)' in feature
    assert 'return _get_setting("DBT_SIDECAR_POOL", "DBT_SIDECAR_POOL", DBT_SIDECAR_POOL)' in quality
    assert (
        'return _get_setting("AIRFLOW_QUEUE_DAILY_SIDECAR", "AIRFLOW_QUEUE_DAILY_SIDECAR", DAILY_SIDECAR_QUEUE)'
        in quality
    )
    assert 'return _get_setting("AIRFLOW_QUEUE_WEATHER_10M", "AIRFLOW_QUEUE_WEATHER_10M", WEATHER_10M_QUEUE)' in weather
    assert 'return _get_setting("DBT_WEATHER_POOL", "DBT_WEATHER_POOL", DBT_WEATHER_POOL)' in weather_refresh
    assert (
        'return _get_setting("AIRFLOW_QUEUE_WEATHER_10M", "AIRFLOW_QUEUE_WEATHER_10M", WEATHER_10M_QUEUE)'
        in weather_refresh
    )
    assert "max_active_runs=1" in weather


def test_dbt_runtime_selectors_and_thread_defaults_are_hotpath_safe():
    selectors = Path("dbt/bikeshare_dbt/selectors.yml").read_text(encoding="utf-8")
    feature = Path("airflow/dags/dbt_feature_build_dag.py").read_text(encoding="utf-8")
    hotpath = Path("airflow/dags/dbt_station_status_hotpath_dag.py").read_text(encoding="utf-8")
    quality = Path("airflow/dags/dbt_quality_hourly_dag.py").read_text(encoding="utf-8")
    diagnostic = Path("airflow/dags/dbt_diagnostic_daily_dag.py").read_text(encoding="utf-8")
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

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

    assert 'get_dbt_threads(_get_setting, "DBT_FEATURE_THREADS")' in feature
    assert 'get_dbt_threads(_get_setting, "DBT_HOTPATH_THREADS")' in hotpath
    assert 'get_dbt_threads(_get_setting, "DBT_SIDECAR_THREADS")' in quality
    assert 'get_dbt_threads(_get_setting, "DBT_SIDECAR_THREADS")' in diagnostic
    assert 'DBT_THREADS: "1"' in compose
    assert 'DBT_HOTPATH_THREADS: "2"' in compose
    assert 'DBT_FEATURE_THREADS: "2"' in compose
    assert 'DBT_WEATHER_THREADS: "2"' in compose
    assert 'DBT_SIDECAR_THREADS: "1"' in compose


def test_dbt_selector_surface_matches_canonical_contract():
    selectors = Path("dbt/bikeshare_dbt/selectors.yml").read_text(encoding="utf-8")
    selector_names = [
        line.strip().split(": ", 1)[1] for line in selectors.splitlines() if line.strip().startswith("- name: ")
    ]

    assert selector_names == [
        "hf_feature_build_models",
        "hf_station_status_hotpath_models",
        "weather_refresh_models",
        "station_topology_daily_models",
        "hf_feature_smoke_tests",
        "hf_station_status_smoke_tests",
        "hourly_quality_gate_tests",
        "daily_deep_quality_tests",
    ]


def test_hotpath_tests_are_retiered_out_of_quality_gate():
    hotpath_unique = Path("dbt/bikeshare_dbt/tests/fct_station_status_unique_grain.sql").read_text(encoding="utf-8")
    enriched_unique = Path("dbt/bikeshare_dbt/tests/int_station_status_enriched_unique_grain.sql").read_text(
        encoding="utf-8"
    )
    weather_coverage = Path(
        "dbt/bikeshare_dbt/tests/int_station_status_enriched_weather_context_coverage.sql"
    ).read_text(encoding="utf-8")

    assert "hf_hotpath_smoke" in hotpath_unique
    assert "quality_gate" not in hotpath_unique
    assert "hf_hotpath_smoke" in enriched_unique
    assert "quality_gate" not in enriched_unique
    assert "quality_gate" in weather_coverage


def test_hotpath_smoke_and_feature_grain_checks_are_window_bounded():
    hotpath_unique = Path("dbt/bikeshare_dbt/tests/fct_station_status_unique_grain.sql").read_text(encoding="utf-8")
    enriched_unique = Path("dbt/bikeshare_dbt/tests/int_station_status_enriched_unique_grain.sql").read_text(
        encoding="utf-8"
    )
    feature_unique = Path("dbt/bikeshare_dbt/tests/feat_station_snapshot_5min_unique_grain.sql").read_text(
        encoding="utf-8"
    )

    for content in (hotpath_unique, enriched_unique):
        assert "runtime_window_start_utc_expr" in content
        assert "runtime_utc_expr('test_window_end_utc')" in content

    assert "runtime_window_start_utc_expr" in feature_unique
    assert "feature_dt_to_utc('dt')" in feature_unique


def test_heavy_feature_latest_assertions_are_retiered_out_of_5min_smoke():
    covers_recent = Path(
        "dbt/bikeshare_dbt/tests/feat_station_snapshot_latest_covers_recent_feature_station_set.sql"
    ).read_text(encoding="utf-8")
    is_latest = Path("dbt/bikeshare_dbt/tests/feat_station_snapshot_latest_is_latest_per_station.sql").read_text(
        encoding="utf-8"
    )
    matches_latest = Path(
        "dbt/bikeshare_dbt/tests/feat_station_snapshot_latest_matches_latest_eligible_feature_set.sql"
    ).read_text(encoding="utf-8")

    for content in (covers_recent, is_latest, matches_latest):
        assert "quality_gate" in content
        assert "hf_smoke" not in content


def test_hourly_quality_gate_excludes_static_dim_and_daily_source_checks():
    dim_date = Path("dbt/bikeshare_dbt/tests/dim_date_contiguous.sql").read_text(encoding="utf-8")
    dim_time_grid = Path("dbt/bikeshare_dbt/tests/dim_time_complete_5min_grid.sql").read_text(encoding="utf-8")
    dim_time_bucket = Path("dbt/bikeshare_dbt/tests/dim_time_bucket_label_consistent.sql").read_text(encoding="utf-8")
    neighbor_defaults = Path("dbt/bikeshare_dbt/tests/feat_station_snapshot_5min_neighbor_defaults.sql").read_text(
        encoding="utf-8"
    )
    targets_match_t30 = Path("dbt/bikeshare_dbt/tests/feat_station_snapshot_5min_targets_match_t30.sql").read_text(
        encoding="utf-8"
    )
    staging_schema = Path("dbt/bikeshare_dbt/models/staging/schema.yml").read_text(encoding="utf-8")

    for content in (dim_date, dim_time_grid, dim_time_bucket, neighbor_defaults, targets_match_t30):
        assert "deep_quality" in content
        assert "quality_gate" not in content

    assert 'tags: ["deep_quality"]' in staging_schema


def test_compose_uses_canonical_dbt_test_selectors():
    base_compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert 'DBT_STATION_STATUS_HOTPATH_TEST_SELECTOR: "hf_station_status_smoke_tests"' in base_compose
    assert 'DBT_FEATURE_BUILD_TEST_SELECTOR: "hf_feature_smoke_tests"' in base_compose
    assert 'DBT_QUALITY_TEST_SELECTOR: "hourly_quality_gate_tests"' in base_compose
    assert 'DBT_DEEP_QUALITY_TEST_SELECTOR: "daily_deep_quality_tests"' in base_compose
    assert 'DBT_FEATURE_BUILD_TEST_SELECTOR: "hf_smoke_tests"' not in base_compose
    assert 'DBT_QUALITY_TEST_SELECTOR: "quality_gate_tests"' not in base_compose


def test_hotpath_build_selector_excludes_static_calendar_models():
    selectors = Path("dbt/bikeshare_dbt/selectors.yml").read_text(encoding="utf-8")

    hotpath_block = selectors.split("- name: hf_station_status_hotpath_models", 1)[1].split(
        "- name: weather_refresh_models", 1
    )[0]

    assert "value: fct_station_status" in hotpath_block
    assert "value: int_station_status_enriched" in hotpath_block
    assert "value: dim_date" not in hotpath_block
    assert "value: dim_time" not in hotpath_block


def test_shared_calendar_and_latest_models_use_low_churn_materializations():
    dim_date = Path("dbt/bikeshare_dbt/models/marts/dim_date.sql").read_text(encoding="utf-8")
    dim_time = Path("dbt/bikeshare_dbt/models/marts/dim_time.sql").read_text(encoding="utf-8")
    feat_latest = Path("dbt/bikeshare_dbt/models/features/feat_station_snapshot_latest.sql").read_text(encoding="utf-8")

    assert "materialized='view'" in dim_date
    assert "materialized='view'" in dim_time
    assert "materialized='incremental'" in feat_latest
    assert "unique_key=['city', 'station_id']" in feat_latest


def test_weather_and_feature_models_encode_runtime_performance_guards():
    dim_weather = Path("dbt/bikeshare_dbt/models/marts/dim_weather.sql").read_text(encoding="utf-8")
    feature_5min = Path("dbt/bikeshare_dbt/models/features/feat_station_snapshot_5min.sql").read_text(encoding="utf-8")

    assert "idx_dim_weather_city_observed_at" in dim_weather
    assert "station_features_windowed as materialized" in feature_5min


def test_runtime_dags_default_to_explicit_build_selectors_instead_of_parent_expansion():
    hotpath_dag = Path("airflow/dags/dbt_station_status_hotpath_dag.py").read_text(encoding="utf-8")
    feature_dag = Path("airflow/dags/dbt_feature_build_dag.py").read_text(encoding="utf-8")
    weather_dag = Path("airflow/dags/dbt_weather_refresh_dag.py").read_text(encoding="utf-8")
    topology_dag = Path("airflow/dags/dbt_station_topology_daily_dag.py").read_text(encoding="utf-8")

    for content, selector_name in (
        (hotpath_dag, "DBT_STATION_STATUS_HOTPATH_SELECTOR"),
        (feature_dag, "DBT_FEATURE_BUILD_SELECTOR"),
        (weather_dag, "DBT_WEATHER_REFRESH_SELECTOR"),
        (topology_dag, "DBT_STATION_TOPOLOGY_SELECTOR"),
    ):
        assert selector_name in content
        assert "selector=None if model_select else build_selector" in content


def test_compose_uses_canonical_dbt_build_selectors():
    base_compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert 'DBT_STATION_STATUS_HOTPATH_SELECTOR: "hf_station_status_hotpath_models"' in base_compose
    assert 'DBT_FEATURE_BUILD_SELECTOR: "hf_feature_build_models"' in base_compose
    assert 'DBT_WEATHER_REFRESH_SELECTOR: "weather_refresh_models"' in base_compose
    assert 'DBT_STATION_TOPOLOGY_SELECTOR: "station_topology_daily_models"' in base_compose
    assert 'DBT_DIAGNOSTIC_FEATURE_BUILD_SELECTOR: "hf_feature_build_models"' in base_compose


def test_five_min_runtime_dags_use_fast_retry_delay():
    gbfs_dag = Path("airflow/dags/gbfs_ingestion_dag.py").read_text(encoding="utf-8")
    hotpath_dag = Path("airflow/dags/dbt_station_status_hotpath_dag.py").read_text(encoding="utf-8")
    feature_dag = Path("airflow/dags/dbt_feature_build_dag.py").read_text(encoding="utf-8")

    assert '"retry_delay": timedelta(seconds=30)' in gbfs_dag
    assert "default_args=five_min_default_args" in gbfs_dag
    assert '"retry_delay": timedelta(seconds=30)' in hotpath_dag
    assert '"retry_delay": timedelta(seconds=30)' in feature_dag
