from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PredictionTargetSpec:
    predict_bikes: bool
    target_name: str
    label_column: str
    paired_target_column: str
    score_column: str
    score_bin_column: str
    actual_t30_column: str


def parse_bool_value(value: str | bool | None, *, default: bool | None = None) -> bool:
    if value is None:
        if default is None:
            raise ValueError("boolean value is required")
        return default
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def target_spec_from_predict_bikes(predict_bikes: bool) -> PredictionTargetSpec:
    if predict_bikes:
        return PredictionTargetSpec(
            predict_bikes=True,
            target_name="bikes",
            label_column="y_stockout_bikes_30",
            paired_target_column="target_bikes_t30",
            score_column="yhat_bikes",
            score_bin_column="yhat_bikes_bin",
            actual_t30_column="bikes_t30",
        )
    return PredictionTargetSpec(
        predict_bikes=False,
        target_name="docks",
        label_column="y_stockout_docks_30",
        paired_target_column="target_docks_t30",
        score_column="yhat_docks",
        score_bin_column="yhat_docks_bin",
        actual_t30_column="docks_t30",
    )


def target_spec_from_name(target_name: str) -> PredictionTargetSpec:
    normalized = target_name.strip().lower()
    if normalized == "bikes":
        return target_spec_from_predict_bikes(True)
    if normalized == "docks":
        return target_spec_from_predict_bikes(False)
    raise ValueError(f"unsupported prediction target: {target_name}")


def target_spec_from_metadata(metadata: Mapping[str, object]) -> PredictionTargetSpec:
    if "predict_bikes" in metadata:
        return target_spec_from_predict_bikes(parse_bool_value(metadata.get("predict_bikes")))

    target_name = metadata.get("target_name")
    if isinstance(target_name, str) and target_name.strip():
        return target_spec_from_name(target_name)

    for key in ("label_column", "label"):
        label = metadata.get(key)
        if label == "y_stockout_bikes_30":
            return target_spec_from_predict_bikes(True)
        if label == "y_stockout_docks_30":
            return target_spec_from_predict_bikes(False)

    raise ValueError("prediction target metadata is missing")
