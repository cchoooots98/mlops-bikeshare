import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import mlflow
import pandas as pd
from sqlalchemy import text

from src.features.postgres_store import PostgresFeatureConfig, create_pg_engine, validate_identifier
from src.model_package import compute_package_sha256, load_package_manifest, write_package_manifest
from src.model_target import parse_bool_value, target_spec_from_predict_bikes
from src.orchestration.dbt_tasks import expand_with_parents, run_dbt_command

REPO_ROOT = Path(__file__).resolve().parents[2]
RETRAIN_RESULT_PREFIX = "RETRAIN_RESULT_JSON::"
MIN_VALID_PR_AUC = 0.55
MAX_OVERFIT_GAP = 0.10
DT_FORMAT = "%Y-%m-%d-%H-%M"
INPUT_TS_FORMAT = "%Y-%m-%d %H:%M"


@dataclass(frozen=True)
class RetrainConfig:
    reason: str
    city: str
    predict_bikes: bool
    model_type: str
    lookback_days: int
    pg_host: str
    pg_port: int
    pg_db: str
    pg_user: str
    pg_password: str
    pg_schema: str
    feature_table: str
    experiment: str
    dbt_project_dir: str
    dbt_profiles_dir: str
    max_feature_age_hours: int
    summary_path: str


def build_model_name(city: str, label: str, model_type: str) -> str:
    return f"{city}_{label}_{model_type}"


def feature_source_name(config: RetrainConfig) -> str:
    return f"{validate_identifier(config.pg_schema)}.{validate_identifier(config.feature_table)}"


def build_pg_config(config: RetrainConfig) -> PostgresFeatureConfig:
    return PostgresFeatureConfig(
        pg_host=config.pg_host,
        pg_port=config.pg_port,
        pg_db=config.pg_db,
        pg_user=config.pg_user,
        pg_password=config.pg_password,
        pg_schema=config.pg_schema,
        training_table=config.feature_table,
    )


def get_latest_feature_dt(config: RetrainConfig) -> datetime:
    engine = create_pg_engine(build_pg_config(config))
    sql = (
        f'SELECT max(dt) AS max_dt FROM "{validate_identifier(config.pg_schema)}".'
        f'"{validate_identifier(config.feature_table)}" WHERE city = :city'
    )
    with engine.connect() as connection:
        max_dt = pd.read_sql_query(text(sql), connection, params={"city": config.city}).iloc[0]["max_dt"]
    if not max_dt:
        raise RuntimeError(f"no features found in {feature_source_name(config)} for city={config.city}")
    return datetime.strptime(max_dt, DT_FORMAT).replace(tzinfo=timezone.utc)


def check_feature_freshness(config: RetrainConfig) -> datetime:
    latest_dt = get_latest_feature_dt(config)
    age = datetime.now(timezone.utc) - latest_dt
    if age > timedelta(hours=config.max_feature_age_hours):
        raise RuntimeError(
            f"feature table {feature_source_name(config)} is stale: latest dt {latest_dt.isoformat()} age={age}"
        )
    return latest_dt


def resolve_retrain_dbt_models(config: RetrainConfig) -> list[str]:
    return [config.feature_table]


def run_dbt_refresh(config: RetrainConfig) -> None:
    select_args = expand_with_parents(resolve_retrain_dbt_models(config))
    run_dbt_command("run", config.dbt_project_dir, config.dbt_profiles_dir, select_args, extra_args=["--fail-fast"])
    run_dbt_command("test", config.dbt_project_dir, config.dbt_profiles_dir, select_args, extra_args=["--fail-fast"])


def parse_training_result(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith("TRAINING_RESULT_JSON::"):
            return json.loads(line.split("::", 1)[1])
    raise RuntimeError("training output did not contain a TRAINING_RESULT_JSON payload")


def run_training_job(config: RetrainConfig, latest_dt: datetime) -> dict:
    target_spec = target_spec_from_predict_bikes(config.predict_bikes)
    train_end = latest_dt.strftime(INPUT_TS_FORMAT)
    train_start = (latest_dt - timedelta(days=config.lookback_days)).strftime(INPUT_TS_FORMAT)
    command = [
        sys.executable,
        "-m",
        "src.training.train",
        "--city",
        config.city,
        "--start",
        train_start,
        "--end",
        train_end,
        "--predict-bikes",
        str(target_spec.predict_bikes).lower(),
        "--model-type",
        config.model_type,
        "--pg-schema",
        config.pg_schema,
        "--feature-table",
        config.feature_table,
        "--experiment",
        config.experiment,
        "--run-reason",
        config.reason,
    ]
    env = os.environ.copy()
    env.update(
        {
            "PGHOST": config.pg_host,
            "PGPORT": str(config.pg_port),
            "PGDATABASE": config.pg_db,
            "PGUSER": config.pg_user,
            "PGPASSWORD": config.pg_password,
        }
    )
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.stdout:
        print(completed.stdout.strip())
    if completed.stderr:
        print(completed.stderr.strip())
    completed.check_returncode()
    summary = parse_training_result(completed.stdout)
    summary["time_start"] = train_start
    summary["time_end"] = train_end
    return summary


def evaluate_candidate(summary: dict) -> None:
    pr_auc_valid = float(summary["pr_auc_valid"])
    overfit_gap = float(summary["overfit_gap"])
    if pr_auc_valid < MIN_VALID_PR_AUC:
        raise RuntimeError(f"candidate rejected: pr_auc_valid={pr_auc_valid:.3f} < {MIN_VALID_PR_AUC:.3f}")
    if overfit_gap > MAX_OVERFIT_GAP:
        raise RuntimeError(f"candidate rejected: overfit_gap={overfit_gap:.3f} > {MAX_OVERFIT_GAP:.3f}")


def write_summary(summary: dict, output_path: str) -> str:
    summary_path = Path(output_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return str(summary_path)


def register_candidate_model(summary: dict, config: RetrainConfig, model_name: str) -> dict:
    from pipelines.register_model import register_model_version

    package_manifest_relpath = ""
    if summary.get("package_dir"):
        package_dir = Path(summary["package_dir"]).resolve()
        package_manifest_relpath = os.path.relpath(
            package_dir.joinpath("package_manifest.json"),
            Path.cwd().resolve(),
        )

    return register_model_version(
        run_id=summary["run_id"],
        model_name=model_name,
        artifact_path=summary.get("model_artifact_path", "model"),
        alias="candidate",
        version_tags={
            "status": "candidate",
            "run_reason": config.reason,
            "label": summary["label"],
            "target_name": summary["target_name"],
            "predict_bikes": str(summary["predict_bikes"]).lower(),
            "feature_source": feature_source_name(config),
            "feature_contract": summary.get("feature_contract", ""),
            "best_threshold": str(summary.get("best_threshold", "")),
            "package_manifest_relpath": package_manifest_relpath,
        },
    )


def update_registered_package_metadata(summary: dict, registration: dict, model_name: str) -> dict[str, str]:
    package_dir = summary.get("package_dir")
    if not package_dir:
        return {}
    manifest = load_package_manifest(package_dir)
    manifest["registered_model_name"] = model_name
    manifest["registered_version"] = registration["version"]
    manifest["aliases"] = ["candidate"]
    manifest_path = write_package_manifest(package_dir, manifest)
    return {
        "package_manifest_relpath": os.path.relpath(Path(manifest_path), Path.cwd().resolve()),
        "package_sha256": compute_package_sha256(package_dir),
    }


def update_registration_package_tags(registration: dict, package_metadata: dict[str, str]) -> None:
    if not package_metadata:
        return
    client = mlflow.tracking.MlflowClient()
    for key, value in package_metadata.items():
        client.set_model_version_tag(registration["model_name"], registration["version"], key, value)


def run_retraining(config: RetrainConfig) -> dict:
    run_dbt_refresh(config)
    check_feature_freshness(config)
    latest_dt = get_latest_feature_dt(config)
    summary = run_training_job(config, latest_dt)
    evaluate_candidate(summary)

    model_name = build_model_name(config.city, summary["label"], config.model_type)
    registration = register_candidate_model(summary, config, model_name)
    package_metadata = update_registered_package_metadata(summary, registration, model_name)
    update_registration_package_tags(registration, package_metadata)

    retrain_summary = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reason": config.reason,
        "city": config.city,
        "predict_bikes": summary["predict_bikes"],
        "target_name": summary["target_name"],
        "label": summary["label"],
        "model_type": config.model_type,
        "lookback_days": config.lookback_days,
        "feature_source": feature_source_name(config),
        "latest_feature_dt_utc": latest_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "next_action": "deploy_candidate",
        "training": summary,
        "registration": registration,
        "package": package_metadata,
    }
    retrain_summary["summary_path"] = write_summary(retrain_summary, config.summary_path)
    print(RETRAIN_RESULT_PREFIX + json.dumps(retrain_summary, sort_keys=True))
    return retrain_summary


def env_or_default(key: str, default: str | None = None) -> str | None:
    value = os.environ.get(key)
    if value is not None and value != "":
        return value
    return default


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline retraining and register a candidate model.")
    parser.add_argument("--reason", required=True, choices=["schedule", "manual", "drift", "post_rollback"])
    parser.add_argument("--city", required=True)
    parser.add_argument("--predict-bikes", default=env_or_default("PREDICT_BIKES", "true"), choices=["true", "false"])
    parser.add_argument("--model-type", required=True, choices=["xgboost", "lightgbm"])
    parser.add_argument("--lookback-days", required=True, type=int)
    parser.add_argument("--pg-host", default=env_or_default("PGHOST"))
    parser.add_argument("--pg-port", default=env_or_default("PGPORT", "5432"), type=int)
    parser.add_argument("--pg-db", default=env_or_default("PGDATABASE"))
    parser.add_argument("--pg-user", default=env_or_default("PGUSER"))
    parser.add_argument("--pg-password", default=env_or_default("PGPASSWORD"))
    parser.add_argument("--pg-schema", default=env_or_default("PGSCHEMA", "analytics"))
    parser.add_argument("--feature-table", default=env_or_default("FEATURE_TABLE", "feat_station_snapshot_5min"))
    parser.add_argument("--experiment", default=env_or_default("TRAINING_EXPERIMENT", "bikeshare-offline-retrain"))
    parser.add_argument("--dbt-project-dir", default=env_or_default("DBT_PROJECT_DIR", "dbt/bikeshare_dbt"))
    parser.add_argument("--dbt-profiles-dir", default=env_or_default("DBT_PROFILES_DIR", "dbt"))
    parser.add_argument("--max-feature-age-hours", default=48, type=int)
    parser.add_argument(
        "--summary-path",
        default=env_or_default("RETRAIN_SUMMARY_PATH", "model_dir/candidates/retrain_summary.json"),
    )
    args = parser.parse_args(argv)

    missing = [name for name in ("pg_host", "pg_db", "pg_user", "pg_password") if getattr(args, name) in {None, ""}]
    if missing:
        raise ValueError(f"missing required Postgres settings: {missing}")
    return args


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    config = RetrainConfig(
        reason=args.reason,
        city=args.city,
        predict_bikes=parse_bool_value(args.predict_bikes, default=True),
        model_type=args.model_type,
        lookback_days=args.lookback_days,
        pg_host=args.pg_host,
        pg_port=args.pg_port,
        pg_db=args.pg_db,
        pg_user=args.pg_user,
        pg_password=args.pg_password,
        pg_schema=args.pg_schema,
        feature_table=args.feature_table,
        experiment=args.experiment,
        dbt_project_dir=args.dbt_project_dir,
        dbt_profiles_dir=args.dbt_profiles_dir,
        max_feature_age_hours=args.max_feature_age_hours,
        summary_path=args.summary_path,
    )
    return run_retraining(config)


if __name__ == "__main__":
    main()
