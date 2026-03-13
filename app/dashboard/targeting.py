from __future__ import annotations

from dataclasses import dataclass

from src.config import endpoint_name, prediction_prefix, quality_prefix, resolve_target_name
from src.model_target import target_spec_from_name


@dataclass(frozen=True)
class DashboardTargetConfig:
    target_name: str
    display_name: str
    label_column: str
    score_column: str
    score_bin_column: str
    endpoint_name: str
    inference_prefix: str
    quality_prefix: str
    section_title: str


def resolve_target_columns(target_name: str) -> dict[str, str]:
    spec = target_spec_from_name(resolve_target_name(target_name=target_name))
    return {
        "target_name": spec.target_name,
        "label_column": spec.label_column,
        "score_column": spec.score_column,
        "score_bin_column": spec.score_bin_column,
    }


def resolve_dashboard_target(*, target_name: str, city: str, environment: str, project_slug: str = "bikeshare") -> DashboardTargetConfig:
    columns = resolve_target_columns(target_name)
    resolved_target_name = columns["target_name"]
    display_name = "Bike stockout" if resolved_target_name == "bikes" else "Dock stockout"
    return DashboardTargetConfig(
        target_name=resolved_target_name,
        display_name=display_name,
        label_column=columns["label_column"],
        score_column=columns["score_column"],
        score_bin_column=columns["score_bin_column"],
        endpoint_name=endpoint_name(target_name=resolved_target_name, environment=environment, project_slug=project_slug),
        inference_prefix=prediction_prefix(city, resolved_target_name),
        quality_prefix=quality_prefix(city, resolved_target_name),
        section_title=f"{display_name} Risk",
    )
