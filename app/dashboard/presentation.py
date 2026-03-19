from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from .contracts import ArtifactLoadResult, FreshnessLoadResult, LoadStatus

_RISK_COLORS = {
    "Critical": "#c1121f",
    "Alert": "#f77f00",
    "Normal": "#2a9d8f",
}

_STATUS_COLORS = {
    "Healthy": "#2a9d8f",
    "Warning": "#f77f00",
    "Critical": "#c1121f",
    "Observed": "#577590",
    "Unavailable": "#577590",
}


@dataclass(frozen=True)
class MetricSpec:
    label: str
    direction: str = "none"
    warning: float | None = None
    critical: float | None = None
    decimals: int = 3
    empty_message: str = "No metric samples available yet."


def critical_threshold_for(threshold: float) -> float:
    return max(threshold, 0.7)


def classify_risk(score: float, *, alert_threshold: float, critical_threshold: float | None = None) -> tuple[str, str]:
    if critical_threshold is not None and score >= critical_threshold:
        return "Critical", _RISK_COLORS["Critical"]
    if score >= alert_threshold:
        return "Alert", _RISK_COLORS["Alert"]
    return "Normal", _RISK_COLORS["Normal"]


def inventory_status_for_station(
    *,
    bikes: object,
    docks: object,
    capacity: object,
    target_name: str,
    stockout_threshold: int = 2,
) -> str:
    inventory = pd.to_numeric(pd.Series([bikes if target_name == "bikes" else docks]), errors="coerce").iloc[0]
    capacity_value = pd.to_numeric(pd.Series([capacity]), errors="coerce").iloc[0]
    if pd.isna(inventory):
        return "Inventory unavailable"
    if inventory <= stockout_threshold:
        return "Stockout now"
    if pd.notna(capacity_value) and capacity_value > 0 and (inventory / capacity_value) <= 0.15:
        return "Low inventory now"
    return "Inventory OK"


def build_station_risk_frame(
    *,
    station_info: pd.DataFrame,
    prediction_result: ArtifactLoadResult,
    target_name: str,
    threshold: float,
) -> pd.DataFrame:
    if prediction_result.status != LoadStatus.OK or station_info.empty:
        return pd.DataFrame()

    merged = station_info.merge(prediction_result.data, on="station_id", how="inner")
    if merged.empty:
        return merged

    critical_threshold = critical_threshold_for(threshold)
    merged = merged.copy()
    merged["risk_level"] = merged["score"].apply(
        lambda score: classify_risk(score, alert_threshold=threshold, critical_threshold=critical_threshold)[0]
    )
    merged["risk_color"] = merged["score"].apply(
        lambda score: classify_risk(score, alert_threshold=threshold, critical_threshold=critical_threshold)[1]
    )
    merged["stockout_probability"] = merged["score"]
    merged["current_status"] = merged.apply(
        lambda row: inventory_status_for_station(
            bikes=row.get("bikes"),
            docks=row.get("docks"),
            capacity=row.get("capacity"),
            target_name=target_name,
        ),
        axis=1,
    )
    merged = merged.sort_values(["score", "station_name", "station_id"], ascending=[False, True, True]).reset_index(drop=True)
    return merged


def resolve_selected_station(
    *,
    station_risk_frame: pd.DataFrame,
    selected_station_id: str | None,
) -> dict[str, object] | None:
    if station_risk_frame.empty:
        return None

    if selected_station_id:
        match = station_risk_frame[station_risk_frame["station_id"].astype(str) == str(selected_station_id)]
        if not match.empty:
            return match.iloc[0].to_dict()

    return station_risk_frame.iloc[0].to_dict()


def station_history_title() -> str:
    return "Station History"


def station_history_context(selected_station: dict[str, object] | None) -> str:
    station_name = str((selected_station or {}).get("station_name") or "No station selected")
    station_id = str((selected_station or {}).get("station_id") or "n/a")
    return f"Station: {station_name} ({station_id}). Forecast horizon: next 30 minutes (UTC)."


def format_utc_label(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Unavailable"
    timestamp = pd.to_datetime(value, format="%Y-%m-%d-%H-%M", utc=True, errors="coerce")
    if pd.isna(timestamp):
        timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return str(value)
    return timestamp.strftime("%Y-%m-%d %H:%M UTC")


def summarize_quality_availability(
    *,
    quality_result: ArtifactLoadResult,
    metric_series_map: dict[str, pd.DataFrame],
) -> tuple[str, str, str]:
    missing_metrics = [metric_name for metric_name, series in metric_series_map.items() if series.empty]

    if quality_result.status == LoadStatus.NO_OBJECTS:
        return (
            "warning",
            "Quality evidence is still maturing.",
            "No quality artifact is available yet. This is expected until the 30-minute label window has matured and the backfill job has run.",
        )
    if quality_result.status == LoadStatus.ALL_SCORES_NULL:
        return (
            "error",
            "Quality artifact is invalid.",
            quality_result.message or "The latest quality shard exists, but all scores are null or invalid.",
        )
    if quality_result.status in {LoadStatus.ACCESS_DENIED, LoadStatus.READ_ERROR, LoadStatus.SCHEMA_ERROR}:
        return (
            "error",
            "Quality artifact could not be trusted.",
            quality_result.message or "The latest quality artifact could not be read or did not match the expected schema.",
        )
    if missing_metrics:
        return (
            "info",
            "CloudWatch quality metrics are not published yet.",
            "A quality artifact exists, but one or more CloudWatch series are still empty. This usually means metric publication is delayed or the metric dimensions do not match the selected target/environment.",
        )
    return (
        "success",
        "Quality data and CloudWatch metrics are available.",
        "The latest mature quality shard and its published monitoring series are both present for this target.",
    )


def classify_metric_status(value: float, spec: MetricSpec) -> tuple[str, str]:
    if spec.direction == "higher":
        if spec.critical is not None and value < spec.critical:
            return "Critical", _STATUS_COLORS["Critical"]
        if spec.warning is not None and value < spec.warning:
            return "Warning", _STATUS_COLORS["Warning"]
        return "Healthy", _STATUS_COLORS["Healthy"]
    if spec.direction == "lower":
        if spec.critical is not None and value > spec.critical:
            return "Critical", _STATUS_COLORS["Critical"]
        if spec.warning is not None and value > spec.warning:
            return "Warning", _STATUS_COLORS["Warning"]
        return "Healthy", _STATUS_COLORS["Healthy"]
    return "Observed", _STATUS_COLORS["Observed"]


def metric_empty_message(
    *,
    metric_name: str,
    spec: MetricSpec | None,
    quality_result: ArtifactLoadResult | None = None,
) -> str:
    if quality_result is not None and metric_name in {
        "PR-AUC-24h",
        "F1-24h",
        "PredictionHeartbeat",
        "ThresholdHitRate-24h",
        "Samples-24h",
    }:
        if quality_result.status == LoadStatus.NO_OBJECTS:
            return "No mature quality shard is available yet for the last 24 hours."
        if quality_result.status == LoadStatus.ALL_SCORES_NULL:
            return "The latest quality shard is invalid because all prediction scores are null or unusable."
    if spec is not None:
        return spec.empty_message
    return "No metric samples available yet."


def _artifact_delay_minutes(*, latest_dt: datetime | None, now_utc: datetime) -> float | None:
    if latest_dt is None:
        return None
    return round((now_utc - latest_dt).total_seconds() / 60, 1)


def _status_from_delay(*, delay_minutes: float | None, sla_minutes: int) -> tuple[str, str]:
    if delay_minutes is None:
        return "Critical", "No timestamp is available."
    if delay_minutes > sla_minutes:
        return "Critical", f"Freshness exceeds the {sla_minutes} minute expectation."
    return "Healthy", f"Freshness is within the {sla_minutes} minute expectation."


def build_data_status_frame(
    *,
    prediction_result: ArtifactLoadResult,
    quality_result: ArtifactLoadResult,
    freshness_result: FreshnessLoadResult,
    prediction_sla_minutes: int,
    quality_sla_minutes: int,
    feature_sla_minutes: int,
) -> pd.DataFrame:
    now_utc = datetime.now(timezone.utc)
    rows: list[dict[str, object]] = []

    prediction_delay = _artifact_delay_minutes(latest_dt=prediction_result.latest_dt, now_utc=now_utc)
    if prediction_result.status == LoadStatus.OK:
        prediction_status, prediction_meaning = _status_from_delay(
            delay_minutes=prediction_delay,
            sla_minutes=prediction_sla_minutes,
        )
    elif prediction_result.status == LoadStatus.NO_OBJECTS:
        prediction_status, prediction_meaning = (
            "Critical",
            "Predictions are missing; operators should verify the serving prediction DAG and endpoint.",
        )
    else:
        prediction_status, prediction_meaning = (
            "Critical",
            prediction_result.message or "Prediction artifact is unreadable or invalid.",
        )
    rows.append(
        {
            "Data source": "Prediction artifact",
            "Last updated (UTC)": prediction_result.latest_dt,
            "Delay (min)": prediction_delay,
            "Status": prediction_status,
            "Expected cadence / SLA": f"Prediction every 15 min; stale after {prediction_sla_minutes} min",
            "Operator meaning": prediction_meaning,
            "Details": prediction_result.message or (prediction_result.latest_key or ""),
        }
    )

    quality_delay = _artifact_delay_minutes(latest_dt=quality_result.latest_dt, now_utc=now_utc)
    if quality_result.status == LoadStatus.OK:
        quality_status, quality_meaning = _status_from_delay(
            delay_minutes=quality_delay,
            sla_minutes=quality_sla_minutes,
        )
    elif quality_result.status == LoadStatus.NO_OBJECTS:
        if prediction_delay is not None and prediction_delay <= quality_sla_minutes:
            quality_status, quality_meaning = (
                "Warning",
                "Quality output is still awaiting label maturity and the scheduled backfill window.",
            )
        else:
            quality_status, quality_meaning = (
                "Critical",
                "Quality artifacts are missing beyond the expected maturity window.",
            )
    else:
        quality_status, quality_meaning = (
            "Critical",
            quality_result.message or "Quality artifact is unreadable or invalid.",
        )
    rows.append(
        {
            "Data source": "Quality artifact",
            "Last updated (UTC)": quality_result.latest_dt,
            "Delay (min)": quality_delay,
            "Status": quality_status,
            "Expected cadence / SLA": (
                "30 min label maturity + 7 min backfill lag; expect within "
                f"{quality_sla_minutes} min"
            ),
            "Operator meaning": quality_meaning,
            "Details": quality_result.message or (quality_result.latest_key or ""),
        }
    )

    freshness = freshness_result.data.copy()
    if not freshness.empty:
        freshness["Last updated (UTC)"] = pd.to_datetime(
            freshness["latest_dt_str"],
            format="%Y-%m-%d-%H-%M",
            errors="coerce",
            utc=True,
        )
        freshness["Delay (min)"] = (
            now_utc - freshness["Last updated (UTC)"]
        ).dt.total_seconds().div(60).round(1)

        def _freshness_row(row: pd.Series) -> tuple[str, str]:
            if row["loader_status"] != "ok":
                return "Critical", row["message"] or "Feature freshness could not be read."
            delay = row["Delay (min)"]
            if pd.isna(delay):
                return "Critical", "Feature freshness has no timestamp."
            if delay > feature_sla_minutes:
                return "Critical", f"Serving features are stale beyond {feature_sla_minutes} minutes."
            return "Healthy", "Serving features are fresh enough for operator use."

        freshness[["Status", "Operator meaning"]] = freshness.apply(
            lambda row: pd.Series(_freshness_row(row)),
            axis=1,
        )
        freshness["Expected cadence / SLA"] = f"Serving features fresh within {feature_sla_minutes} min"
        freshness["Details"] = freshness["message"].fillna("")
        freshness = freshness.rename(columns={"source": "Data source"})
        rows.extend(
            freshness[
                [
                    "Data source",
                    "Last updated (UTC)",
                    "Delay (min)",
                    "Status",
                    "Expected cadence / SLA",
                    "Operator meaning",
                    "Details",
                ]
            ].to_dict("records")
        )

    return pd.DataFrame(rows)
