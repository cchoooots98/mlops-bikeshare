from __future__ import annotations

from dataclasses import dataclass

from src.config import deployment_state_path
from src.model_package import load_deployment_state, load_package_manifest


@dataclass(frozen=True)
class DashboardModelMetadata:
    status: str
    message: str
    model_name: str | None
    run_id: str | None
    deployment_state_path: str | None
    package_manifest_path: str | None
    threshold: float | None

    @property
    def display_version(self) -> str:
        if self.run_id:
            return self.run_id
        if self.model_name:
            return self.model_name
        return "unavailable"


def load_dashboard_model_metadata(
    *,
    target_name: str,
    environment: str,
    fallback_model_version: str | None = None,
) -> DashboardModelMetadata:
    try:
        state_path = deployment_state_path(target_name=target_name, environment=environment)
        state = load_deployment_state(state_path)
        manifest = load_package_manifest(state["package_manifest_path"])
        return DashboardModelMetadata(
            status="ok",
            message="",
            model_name=str(manifest.get("model_name") or "") or None,
            run_id=str(manifest.get("run_id") or "") or None,
            deployment_state_path=str(state_path),
            package_manifest_path=str(state["package_manifest_path"]),
            threshold=float(manifest["best_threshold"]) if manifest.get("best_threshold") is not None else None,
        )
    except Exception as exc:
        return DashboardModelMetadata(
            status="read_error",
            message=f"Unable to load deployment metadata: {exc}",
            model_name=(
                fallback_model_version if fallback_model_version and fallback_model_version != "unknown" else None
            ),
            run_id=None,
            deployment_state_path=None,
            package_manifest_path=None,
            threshold=None,
        )
