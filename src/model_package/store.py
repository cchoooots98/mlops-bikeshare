from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PACKAGE_LAYOUT_VERSION = "2"
DEPLOYMENT_STATE_VERSION = "2"
PACKAGE_MANIFEST_FILENAME = "package_manifest.json"
MODEL_DIRNAME = "model"
ARTIFACTS_DIRNAME = "artifacts"
DEFAULT_PACKAGE_ROOT = Path("model_dir") / "packages"
DEFAULT_BIKES_PACKAGE_ROOT = DEFAULT_PACKAGE_ROOT / "bikes"
DEFAULT_DOCKS_PACKAGE_ROOT = DEFAULT_PACKAGE_ROOT / "docks"
DEFAULT_DEPLOYMENT_STATE_PATH = Path("model_dir") / "deployments" / "bikes" / "local.json"
PORTABLE_PATH_ANCHOR = "model_dir"
SUPPORTED_CONTRACT_VERSIONS = {"1", "2"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_SLUG_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_REQUIRED_PACKAGE_FIELDS = {
    "package_layout_version",
    "model_name",
    "run_id",
    "model_type",
    "predict_bikes",
    "target_name",
    "label_column",
    "paired_target_column",
    "score_column",
    "score_bin_column",
    "actual_t30_column",
    "best_threshold",
    "pr_auc_valid",
    "overfit_gap",
    "feature_contract_version",
    "feature_columns",
    "feature_source",
    "time_start",
    "time_end",
    "registered_model_name",
    "registered_version",
    "aliases",
    "paths",
}
_REQUIRED_DEPLOYMENT_FIELDS = {
    "deployment_state_version",
    "environment",
    "predict_bikes",
    "target_name",
    "package_dir",
    "package_manifest_path",
    "model_name",
    "run_id",
    "updated_at_utc",
    "source",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slugify(value: str) -> str:
    normalized = _SLUG_PATTERN.sub("-", value.strip())
    return normalized.strip("-") or "package"


def _normalize_path_parts(path_like: str | os.PathLike[str]) -> list[str]:
    text = str(path_like).replace("\\", "/").strip()
    return [part for part in text.split("/") if part not in {"", "."}]


def _extract_anchor_suffix(path_like: str | os.PathLike[str]) -> Path | None:
    parts = _normalize_path_parts(path_like)
    lowered = [part.lower() for part in parts]
    if PORTABLE_PATH_ANCHOR not in lowered:
        return None
    anchor_index = lowered.index(PORTABLE_PATH_ANCHOR)
    return Path(*parts[anchor_index:])


def _infer_project_root(reference_path: str | os.PathLike[str]) -> Path:
    resolved = Path(reference_path).resolve()
    lowered = [part.lower() for part in resolved.parts]
    if PORTABLE_PATH_ANCHOR not in lowered:
        return PROJECT_ROOT
    anchor_index = lowered.index(PORTABLE_PATH_ANCHOR)
    return Path(*resolved.parts[:anchor_index]) if anchor_index > 0 else PROJECT_ROOT


def _resolve_runtime_path(path_like: str | os.PathLike[str], *, project_root: Path) -> Path:
    candidate = Path(path_like)
    if candidate.exists():
        return candidate.resolve()

    suffix = _extract_anchor_suffix(path_like)
    if suffix is not None:
        return (project_root / suffix).resolve()

    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()


def _portable_path(path_like: str | os.PathLike[str], *, project_root: Path) -> str:
    resolved = _resolve_runtime_path(path_like, project_root=project_root)
    suffix = _extract_anchor_suffix(resolved)
    if suffix is not None:
        return suffix.as_posix()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _build_portable_manifest_paths(package_dir: str | os.PathLike[str]) -> dict[str, str]:
    package_path = Path(package_dir).resolve()
    project_root = _infer_project_root(package_path)
    return {
        "package_dir": _portable_path(package_path, project_root=project_root),
        "model_dir": _portable_path(package_path / MODEL_DIRNAME, project_root=project_root),
        "package_manifest_path": _portable_path(package_path / PACKAGE_MANIFEST_FILENAME, project_root=project_root),
        "artifacts_dir": _portable_path(package_path / ARTIFACTS_DIRNAME, project_root=project_root),
    }


def _resolve_manifest_paths(paths: Mapping[str, Any], *, manifest_path: Path) -> dict[str, str]:
    project_root = _infer_project_root(manifest_path)
    package_dir = _resolve_runtime_path(paths["package_dir"], project_root=project_root)
    return {
        "package_dir": str(package_dir),
        "model_dir": str(_resolve_runtime_path(paths.get("model_dir", package_dir / MODEL_DIRNAME), project_root=project_root)),
        "package_manifest_path": str(manifest_path.resolve()),
        "artifacts_dir": str(
            _resolve_runtime_path(paths.get("artifacts_dir", package_dir / ARTIFACTS_DIRNAME), project_root=project_root)
        ),
    }


def _normalize_manifest_for_storage(
    manifest: Mapping[str, Any],
    *,
    package_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    normalized = dict(manifest)
    normalized["package_layout_version"] = PACKAGE_LAYOUT_VERSION
    normalized["paths"] = _build_portable_manifest_paths(package_dir)
    return normalized


def _normalize_deployment_state_for_storage(
    state: Mapping[str, Any],
    *,
    state_path: str | os.PathLike[str],
) -> dict[str, Any]:
    project_root = _infer_project_root(Path(state_path).resolve())
    package_dir = _resolve_runtime_path(state["package_dir"], project_root=project_root)
    manifest_value = state.get("package_manifest_path") or package_dir / PACKAGE_MANIFEST_FILENAME
    manifest_path = _resolve_runtime_path(manifest_value, project_root=project_root)

    normalized = dict(state)
    normalized["deployment_state_version"] = DEPLOYMENT_STATE_VERSION
    normalized["package_dir"] = _portable_path(package_dir, project_root=project_root)
    normalized["package_manifest_path"] = _portable_path(manifest_path, project_root=project_root)
    return normalized


def write_json_file(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> str:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(file_path.resolve())


def ensure_package_dir(model_name: str, run_id: str, root_dir: str | os.PathLike[str] | None = None) -> Path:
    base_dir = Path(root_dir) if root_dir is not None else DEFAULT_PACKAGE_ROOT
    package_dir = base_dir / _slugify(model_name) / _slugify(run_id)
    (package_dir / MODEL_DIRNAME).mkdir(parents=True, exist_ok=True)
    (package_dir / ARTIFACTS_DIRNAME).mkdir(parents=True, exist_ok=True)
    return package_dir


def default_package_root_for_target(target_name: str) -> Path:
    normalized = target_name.strip().lower()
    if normalized == "bikes":
        return DEFAULT_BIKES_PACKAGE_ROOT
    if normalized == "docks":
        return DEFAULT_DOCKS_PACKAGE_ROOT
    raise ValueError(f"unsupported package target: {target_name}")


def build_package_manifest(summary: Mapping[str, Any], package_dir: str | os.PathLike[str]) -> dict[str, Any]:
    manifest = {
        "package_layout_version": PACKAGE_LAYOUT_VERSION,
        "created_at_utc": _utc_now(),
        "model_name": summary["model_name"],
        "run_id": summary["run_id"],
        "model_type": summary["model_type"],
        "predict_bikes": bool(summary["predict_bikes"]),
        "target_name": summary["target_name"],
        "label_column": summary["label_column"],
        "paired_target_column": summary["paired_target_column"],
        "score_column": summary["score_column"],
        "score_bin_column": summary["score_bin_column"],
        "actual_t30_column": summary["actual_t30_column"],
        "best_threshold": float(summary["best_threshold"]),
        "pr_auc_valid": float(summary["pr_auc_valid"]),
        "overfit_gap": float(summary["overfit_gap"]),
        "feature_contract_version": summary["feature_contract"],
        "feature_columns": list(summary["features"]),
        "feature_source": summary["feature_source"],
        "city": summary["city"],
        "time_start": summary["time_start"],
        "time_end": summary["time_end"],
        "train_end_dt": summary["train_end_dt"],
        "valid_start_dt": summary["valid_start_dt"],
        "registered_model_name": summary.get("registered_model_name"),
        "registered_version": summary.get("registered_version"),
        "aliases": list(summary.get("aliases", [])),
        "paths": _build_portable_manifest_paths(package_dir),
    }
    validate_package_manifest(manifest)
    return manifest


def validate_package_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if str(manifest.get("package_layout_version")) not in SUPPORTED_CONTRACT_VERSIONS:
        raise ValueError(f"unsupported package layout version: {manifest.get('package_layout_version')}")
    missing = sorted(_REQUIRED_PACKAGE_FIELDS.difference(manifest.keys()))
    if missing:
        raise ValueError(f"package manifest missing required fields: {missing}")

    feature_columns = manifest.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("package manifest feature_columns must be a non-empty list")
    aliases = manifest.get("aliases")
    if not isinstance(aliases, list):
        raise ValueError("package manifest aliases must be a list")
    paths = manifest.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("package manifest paths must be an object")
    for key in ("package_dir", "model_dir", "package_manifest_path", "artifacts_dir"):
        value = paths.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"package manifest paths.{key} must be a non-empty string")
    return dict(manifest)


def resolve_package_manifest_path(package_dir_or_manifest: str | os.PathLike[str]) -> Path:
    candidate = _resolve_runtime_path(package_dir_or_manifest, project_root=PROJECT_ROOT)
    if candidate.is_dir():
        candidate = candidate / PACKAGE_MANIFEST_FILENAME
    if candidate.name != PACKAGE_MANIFEST_FILENAME:
        raise ValueError(f"expected package dir or {PACKAGE_MANIFEST_FILENAME}: {candidate}")
    return candidate


def load_package_manifest(package_dir_or_manifest: str | os.PathLike[str]) -> dict[str, Any]:
    manifest_path = resolve_package_manifest_path(package_dir_or_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validated = validate_package_manifest(manifest)
    resolved = dict(validated)
    resolved["paths"] = _resolve_manifest_paths(validated["paths"], manifest_path=manifest_path)
    return resolved


def write_package_manifest(package_dir: str | os.PathLike[str], manifest: Mapping[str, Any]) -> str:
    normalized = _normalize_manifest_for_storage(manifest, package_dir=package_dir)
    validate_package_manifest(normalized)
    return write_json_file(Path(package_dir) / PACKAGE_MANIFEST_FILENAME, normalized)


def compute_package_sha256(package_dir: str | os.PathLike[str]) -> str:
    package_path = _resolve_runtime_path(package_dir, project_root=PROJECT_ROOT)
    digest = hashlib.sha256()
    if not package_path.exists():
        raise FileNotFoundError(f"package directory does not exist: {package_path}")
    for file_path in sorted(path for path in package_path.rglob("*") if path.is_file()):
        relative = file_path.relative_to(package_path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_model_card_text(summary: Mapping[str, Any], owner: str = "MLOps Team", horizon: int = 30) -> str:
    features = summary.get("features", [])
    target_name = str(summary.get("target_name") or "bikes or docks")
    model_name = str(summary.get("model_name") or f"{summary['model_type']}-{summary['label']}")
    feature_source = str(summary.get("feature_source", "analytics.feat_station_snapshot_5min"))
    feature_contract = str(summary.get("feature_contract", summary.get("feature_contract_version", "unknown")))
    experiment = str(summary.get("experiment", "bikeshare-step4"))
    model_artifact_path = str(summary.get("model_artifact_path", "model"))
    return (
        f"# Model Card - {model_name}\n\n"
        "## Overview\n"
        f"- Use case: Predict short-term ({horizon} min) stockout risk for {target_name} at bikeshare stations.\n"
        "- Business objective: Support proactive rebalancing before a station hits low inventory or low dock capacity.\n"
        f"- Owner: {owner}\n"
        f"- Date (UTC): {_utc_now()}\n\n"
        "## Data\n"
        f"- City: {summary['city']}\n"
        f"- Time window (UTC): {summary['time_start']} -> {summary['time_end']}\n"
        f"- Sample sizes: train={summary['n_train']}, valid={summary['n_valid']}\n"
        f"- Feature source: `{feature_source}`\n"
        f"- Feature contract: `{feature_contract}`\n"
        f"- Feature list (first 10): {', '.join(features[:10])} (total {len(features)})\n"
        f"- Labels: {summary['label']}\n\n"
        "## Modeling\n"
        f"- Model name: {model_name}\n"
        f"- Algorithm: {summary['model_type']}\n"
        f"- Primary metric: PR-AUC (validation) = {float(summary['pr_auc_valid']):.3f}\n"
        f"- Train PR-AUC: {float(summary['pr_auc_train']):.3f}; Overfit gap: {float(summary['overfit_gap']):.3f} (target < 0.10)\n"
        f"- Threshold (F-beta, beta={float(summary['beta']):.1f}): {float(summary['best_threshold']):.2f}\n"
        f"  - Precision={float(summary['best_precision']):.3f}, Recall={float(summary['best_recall']):.3f}, F-beta={float(summary['best_fbeta']):.3f}\n\n"
        "## Assumptions And Limitations\n"
        "- Offline training consumes dbt-owned Postgres feature tables and the Python schema contract in `src/features/schema.py`.\n"
        "- Neighbor features come from the warehouse radius-based neighbor graph, not a runtime BallTree rebuild.\n"
        "- Labels are considered mature only when the full future horizon is present in the offline feature table.\n"
        "- Temporal validation is strictly later than training and separated by an anti-leakage gap.\n\n"
        "## Monitoring Plan\n"
        "- Track PR-AUC, F1, PSI, input freshness, and serving latency after candidate deployment.\n"
        "- Retraining should create a new candidate and pass staging admission checks before production promotion.\n\n"
        "## Reproducibility\n"
        f"- MLflow experiment: {experiment}\n"
        f"- Run ID: {summary['run_id']}\n"
        f"- Serving artifact path: `{model_artifact_path}`\n"
        "- Training code: `src/training/train.py`\n"
        "- Evaluation code: `src/training/eval.py`\n"
    )


def build_deployment_state(
    package_dir: str | os.PathLike[str],
    manifest: Mapping[str, Any],
    *,
    environment: str = "local",
    source: str = "manual",
    endpoint_name: str | None = None,
) -> dict[str, Any]:
    package_path = Path(package_dir).resolve()
    state = {
        "deployment_state_version": DEPLOYMENT_STATE_VERSION,
        "environment": environment,
        "predict_bikes": bool(manifest["predict_bikes"]),
        "target_name": manifest["target_name"],
        "package_dir": _portable_path(package_path, project_root=_infer_project_root(package_path)),
        "package_manifest_path": _portable_path(
            resolve_package_manifest_path(package_path),
            project_root=_infer_project_root(package_path),
        ),
        "model_name": manifest["model_name"],
        "run_id": manifest["run_id"],
        "registered_model_name": manifest.get("registered_model_name"),
        "registered_version": manifest.get("registered_version"),
        "aliases": list(manifest.get("aliases", [])),
        "updated_at_utc": _utc_now(),
        "source": source,
    }
    if endpoint_name:
        state["endpoint_name"] = endpoint_name
    validate_deployment_state(state)
    return state


def validate_deployment_state(state: Mapping[str, Any]) -> dict[str, Any]:
    if str(state.get("deployment_state_version")) not in SUPPORTED_CONTRACT_VERSIONS:
        raise ValueError(f"unsupported deployment state version: {state.get('deployment_state_version')}")
    missing = sorted(_REQUIRED_DEPLOYMENT_FIELDS.difference(state.keys()))
    if missing:
        raise ValueError(f"deployment state missing required fields: {missing}")
    return dict(state)


def write_deployment_state(path: str | os.PathLike[str], state: Mapping[str, Any]) -> str:
    normalized = _normalize_deployment_state_for_storage(state, state_path=path)
    validate_deployment_state(normalized)
    return write_json_file(path, normalized)


def load_deployment_state(path: str | os.PathLike[str]) -> dict[str, Any]:
    state_path = _resolve_runtime_path(path, project_root=PROJECT_ROOT)
    if not state_path.exists():
        raise FileNotFoundError(f"deployment state does not exist: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    validated = validate_deployment_state(state)
    project_root = _infer_project_root(state_path)
    resolved = dict(validated)
    resolved["package_dir"] = str(_resolve_runtime_path(validated["package_dir"], project_root=project_root))
    resolved["package_manifest_path"] = str(
        _resolve_runtime_path(validated["package_manifest_path"], project_root=project_root)
    )
    return resolved


def activate_package(
    package_dir: str | os.PathLike[str],
    deployment_state_path: str | os.PathLike[str] | None = None,
    *,
    environment: str = "local",
    source: str = "manual",
) -> str:
    manifest = load_package_manifest(package_dir)
    state = build_deployment_state(package_dir, manifest, environment=environment, source=source)
    target_path = deployment_state_path or DEFAULT_DEPLOYMENT_STATE_PATH
    return write_deployment_state(target_path, state)


def resolve_active_package_dir(
    *,
    model_package_dir: str | os.PathLike[str] | None = None,
    deployment_state_path: str | os.PathLike[str] | None = None,
) -> Path:
    if model_package_dir:
        return _resolve_runtime_path(model_package_dir, project_root=PROJECT_ROOT)
    state_path = Path(deployment_state_path or DEFAULT_DEPLOYMENT_STATE_PATH)
    state = load_deployment_state(state_path)
    return Path(state["package_dir"]).resolve()
