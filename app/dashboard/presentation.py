from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

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
    summary: str = "latest"
    empty_message: str = "No metric samples available yet."


@dataclass(frozen=True)
class LatencyPolicy:
    label: str
    cadence_minutes: int
    dt_offset_minutes: int
    availability_lag_minutes: int
    warning_excess_minutes: int
    critical_excess_minutes: int

    @property
    def natural_lag_upper_minutes(self) -> int:
        return self.availability_lag_minutes + self.cadence_minutes


@dataclass(frozen=True)
class LatencyAssessment:
    expected_dt: datetime
    observed_delay_minutes: float | None
    expected_delay_minutes: float
    excess_delay_minutes: float | None
    status: str
    operator_meaning: str


@dataclass(frozen=True)
class RawAgePolicy:
    label: str
    cadence_minutes: int
    warning_age_minutes: int
    critical_age_minutes: int


@dataclass(frozen=True)
class RawAgeAssessment:
    observed_delay_minutes: float | None
    status: str
    operator_meaning: str


SOURCE_FRESHNESS_POLICY = RawAgePolicy(
    label="Source freshness",
    cadence_minutes=5,
    warning_age_minutes=10,
    critical_age_minutes=20,
)
FEATURE_FRESHNESS_POLICY = LatencyPolicy(
    label="Feature freshness",
    cadence_minutes=5,
    dt_offset_minutes=0,
    availability_lag_minutes=4,
    warning_excess_minutes=5,
    critical_excess_minutes=10,
)
PREDICTION_ARTIFACT_POLICY = LatencyPolicy(
    label="Prediction artifact",
    cadence_minutes=15,
    dt_offset_minutes=10,
    availability_lag_minutes=6,
    warning_excess_minutes=15,
    critical_excess_minutes=30,
)
QUALITY_ARTIFACT_POLICY = LatencyPolicy(
    label="Quality artifact",
    cadence_minutes=15,
    dt_offset_minutes=10,
    availability_lag_minutes=43,
    warning_excess_minutes=15,
    critical_excess_minutes=30,
)


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
    inventory_column = "bikes" if target_name == "bikes" else "docks"
    merged["_inventory_sort"] = pd.to_numeric(merged[inventory_column], errors="coerce").fillna(-1)
    merged["_capacity_sort"] = pd.to_numeric(merged["capacity"], errors="coerce").fillna(-1)
    merged = (
        merged.sort_values(
            ["score", "_inventory_sort", "_capacity_sort", "station_name", "station_id"],
            ascending=[False, True, False, True, True],
        )
        .drop(columns=["_inventory_sort", "_capacity_sort"])
        .reset_index(drop=True)
    )
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
    now_utc = datetime.now(timezone.utc)

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
            quality_result.message
            or "The latest quality artifact could not be read or did not match the expected schema.",
        )
    if (
        quality_result.status == LoadStatus.OK
        and quality_result.latest_dt is not None
        and quality_result.latest_dt < now_utc - timedelta(hours=24)
    ):
        return (
            "warning",
            "Quality artifact is stale for the 24-hour window.",
            "The latest quality artifact for this target is older than 24 hours, so the dashboard has no recent quality evidence to summarize. Check the quality backfill job before investigating CloudWatch dimensions.",
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
        if (
            quality_result.status == LoadStatus.OK
            and quality_result.latest_dt is not None
            and quality_result.latest_dt < datetime.now(timezone.utc) - timedelta(hours=24)
        ):
            return (
                "No metric samples in the last 24 hours because the latest quality shard for this target is from "
                f"{format_utc_label(quality_result.latest_dt)}."
            )
    if spec is not None:
        return spec.empty_message
    return "No metric samples available yet."


def _artifact_delay_minutes(*, latest_dt: datetime | None, now_utc: datetime) -> float | None:
    if latest_dt is None:
        return None
    return round((now_utc - latest_dt).total_seconds() / 60, 1)


def _parse_dashboard_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, format="%Y-%m-%d-%H-%M", utc=True, errors="coerce")
    if pd.isna(timestamp):
        timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    return timestamp


def _floor_to_schedule(*, value: datetime, cadence_minutes: int, offset_minutes: int) -> datetime:
    origin = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=offset_minutes)
    cadence_seconds = cadence_minutes * 60
    elapsed_seconds = (value - origin).total_seconds()
    steps = int(elapsed_seconds // cadence_seconds)
    return origin + timedelta(seconds=steps * cadence_seconds)


def expected_latest_dt_for_policy(*, now_utc: datetime, policy: LatencyPolicy) -> datetime:
    return _floor_to_schedule(
        value=now_utc - timedelta(minutes=policy.availability_lag_minutes),
        cadence_minutes=policy.cadence_minutes,
        offset_minutes=policy.dt_offset_minutes,
    )


def schedule_expectation_label(policy: LatencyPolicy) -> str:
    warning_cycles = max(1, (policy.warning_excess_minutes + policy.cadence_minutes - 1) // policy.cadence_minutes)
    critical_cycles = max(1, (policy.critical_excess_minutes + policy.cadence_minutes - 1) // policy.cadence_minutes)
    return (
        f"{policy.label} every {policy.cadence_minutes} min; natural lag "
        f"{policy.availability_lag_minutes}-{policy.natural_lag_upper_minutes} min; "
        f"warning after {warning_cycles} missed cycle"
        f"{'' if warning_cycles == 1 else 's'} ({policy.warning_excess_minutes}+ min behind schedule), "
        f"critical after {critical_cycles} missed cycle"
        f"{'' if critical_cycles == 1 else 's'} ({policy.critical_excess_minutes}+ min behind schedule)"
    )


def assess_schedule_freshness(
    *,
    latest_dt: datetime | None,
    policy: LatencyPolicy,
    now_utc: datetime,
    upstream_latest_dt: datetime | None = None,
    upstream_label: str | None = None,
) -> LatencyAssessment:
    scheduled_expected_dt = expected_latest_dt_for_policy(now_utc=now_utc, policy=policy)
    expected_dt = scheduled_expected_dt
    if upstream_latest_dt is not None and upstream_latest_dt < expected_dt:
        expected_dt = upstream_latest_dt
    expected_delay_minutes = round((now_utc - expected_dt).total_seconds() / 60, 1)
    scheduled_delay_minutes = round((now_utc - scheduled_expected_dt).total_seconds() / 60, 1)
    observed_delay_minutes = _artifact_delay_minutes(latest_dt=latest_dt, now_utc=now_utc)
    capped_by_upstream = upstream_latest_dt is not None and expected_dt != scheduled_expected_dt
    if latest_dt is None:
        operator_meaning = "No timestamp is available for schedule-aware freshness checks."
        if capped_by_upstream and upstream_label is not None:
            operator_meaning += (
                f" The schedule would naturally imply {scheduled_delay_minutes:.1f} min of lag, "
                f"but the effective expectation is capped by latest {upstream_label.lower()} "
                f"at {format_utc_label(upstream_latest_dt)}."
            )
        return LatencyAssessment(
            expected_dt=expected_dt,
            observed_delay_minutes=None,
            expected_delay_minutes=expected_delay_minutes,
            excess_delay_minutes=None,
            status="Critical",
            operator_meaning=operator_meaning,
        )

    excess_delay_minutes = round(max(0.0, (expected_dt - latest_dt).total_seconds() / 60), 1)
    missed_cycles = max(1, int(excess_delay_minutes // policy.cadence_minutes)) if excess_delay_minutes > 0 else 0
    if excess_delay_minutes >= policy.critical_excess_minutes:
        status = "Critical"
        summary = (
            f"has missed at least {missed_cycles} scheduled cycle"
            f"{'' if missed_cycles == 1 else 's'}"
        )
    elif excess_delay_minutes >= policy.warning_excess_minutes:
        status = "Warning"
        summary = (
            f"has missed at least {missed_cycles} scheduled cycle"
            f"{'' if missed_cycles == 1 else 's'}"
        )
    else:
        status = "Healthy"
        summary = "matches the current schedule"

    if capped_by_upstream and upstream_label is not None:
        operator_meaning = (
            f"{policy.label} {summary}. Observed lag is {observed_delay_minutes:.1f} min; "
            f"the schedule naturally implies {scheduled_delay_minutes:.1f} min, but the effective expectation "
            f"is capped by latest {upstream_label.lower()} at {format_utc_label(upstream_latest_dt)} "
            f"for an expected lag of {expected_delay_minutes:.1f} min."
        )
    else:
        operator_meaning = (
            f"{policy.label} {summary}. Observed lag is {observed_delay_minutes:.1f} min; "
            f"the schedule naturally implies {expected_delay_minutes:.1f} min at this moment."
        )
    if excess_delay_minutes > 0:
        operator_meaning += f" Excess lag vs schedule: {excess_delay_minutes:.1f} min."

    return LatencyAssessment(
        expected_dt=expected_dt,
        observed_delay_minutes=observed_delay_minutes,
        expected_delay_minutes=expected_delay_minutes,
        excess_delay_minutes=excess_delay_minutes,
        status=status,
        operator_meaning=operator_meaning,
    )


def source_expectation_label(policy: RawAgePolicy) -> str:
    return (
        f"{policy.label} should advance every {policy.cadence_minutes} min; "
        f"warning if no new source bucket arrives for {policy.warning_age_minutes}+ min, "
        f"critical after {policy.critical_age_minutes}+ min"
    )


def assess_raw_age_freshness(
    *,
    latest_dt: datetime | None,
    policy: RawAgePolicy,
    now_utc: datetime,
) -> RawAgeAssessment:
    observed_delay_minutes = _artifact_delay_minutes(latest_dt=latest_dt, now_utc=now_utc)
    if observed_delay_minutes is None:
        return RawAgeAssessment(
            observed_delay_minutes=None,
            status="Critical",
            operator_meaning="No source snapshot timestamp is available.",
        )
    if observed_delay_minutes >= policy.critical_age_minutes:
        status = "Critical"
        summary = "has been stale for multiple upstream cycles"
    elif observed_delay_minutes >= policy.warning_age_minutes:
        status = "Warning"
        summary = "is older than the upstream freshness target"
    else:
        status = "Healthy"
        summary = "is advancing within the upstream freshness target"
    return RawAgeAssessment(
        observed_delay_minutes=observed_delay_minutes,
        status=status,
        operator_meaning=(
            f"{policy.label} {summary}. Latest source bucket age is {observed_delay_minutes:.1f} min."
        ),
    )


def build_data_status_frame(
    *,
    prediction_result: ArtifactLoadResult,
    quality_result: ArtifactLoadResult,
    freshness_result: FreshnessLoadResult,
    prediction_sla_minutes: int | None = None,
    quality_sla_minutes: int | None = None,
    feature_sla_minutes: int | None = None,
    now_utc: datetime | None = None,
) -> pd.DataFrame:
    now_utc = now_utc or datetime.now(timezone.utc)
    rows: list[dict[str, object]] = []
    freshness = freshness_result.data.copy()
    latest_source_dt: datetime | None = None
    latest_feature_dt: datetime | None = None

    if not freshness.empty:
        freshness["Last updated (UTC)"] = freshness["latest_dt_str"].apply(_parse_dashboard_timestamp)
        freshness["Delay (min)"] = (now_utc - freshness["Last updated (UTC)"]).dt.total_seconds().div(60).round(1)
        freshness["Expected lag (min)"] = float("nan")
        freshness["Excess lag (min)"] = float("nan")
        freshness["Expected cadence / SLA"] = ""
        if "freshness_type" not in freshness.columns:
            freshness["freshness_type"] = "feature"
        else:
            freshness["freshness_type"] = freshness["freshness_type"].fillna("feature").astype(str)

        ok_freshness = freshness[freshness["loader_status"] == "ok"]
        if not ok_freshness.empty:
            source_candidates = ok_freshness.loc[ok_freshness["freshness_type"] == "source", "Last updated (UTC)"]
            feature_candidates = ok_freshness.loc[ok_freshness["freshness_type"] == "feature", "Last updated (UTC)"]
            if not source_candidates.empty:
                latest_source_value = source_candidates.max()
                if not pd.isna(latest_source_value):
                    latest_source_dt = pd.Timestamp(latest_source_value).to_pydatetime()
            if not feature_candidates.empty:
                latest_feature_value = feature_candidates.max()
                if not pd.isna(latest_feature_value):
                    latest_feature_dt = pd.Timestamp(latest_feature_value).to_pydatetime()

    prediction_assessment = assess_schedule_freshness(
        latest_dt=prediction_result.latest_dt,
        policy=PREDICTION_ARTIFACT_POLICY,
        now_utc=now_utc,
        upstream_latest_dt=latest_feature_dt,
        upstream_label="Feature freshness",
    )
    if prediction_result.status == LoadStatus.OK:
        prediction_status = prediction_assessment.status
        prediction_meaning = prediction_assessment.operator_meaning
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
            "Delay (min)": prediction_assessment.observed_delay_minutes,
            "Expected lag (min)": prediction_assessment.expected_delay_minutes,
            "Excess lag (min)": prediction_assessment.excess_delay_minutes,
            "Status": prediction_status,
            "Expected cadence / SLA": schedule_expectation_label(PREDICTION_ARTIFACT_POLICY),
            "Operator meaning": prediction_meaning,
            "Details": prediction_result.message or (prediction_result.latest_key or ""),
        }
    )

    quality_assessment = assess_schedule_freshness(
        latest_dt=quality_result.latest_dt,
        policy=QUALITY_ARTIFACT_POLICY,
        now_utc=now_utc,
        upstream_latest_dt=prediction_result.latest_dt,
        upstream_label="Prediction artifact",
    )
    if quality_result.status == LoadStatus.OK:
        quality_status = quality_assessment.status
        quality_meaning = quality_assessment.operator_meaning
    elif quality_result.status == LoadStatus.NO_OBJECTS:
        if (
            prediction_result.latest_dt is not None
            and prediction_assessment.observed_delay_minutes is not None
            and prediction_assessment.observed_delay_minutes <= QUALITY_ARTIFACT_POLICY.natural_lag_upper_minutes
        ):
            quality_status, quality_meaning = (
                "Healthy",
                "No quality artifact is expected yet because predictions are still within the 30-minute label maturity plus backfill window.",
            )
        else:
            quality_status, quality_meaning = (
                "Critical",
                "Quality artifacts are missing beyond the natural maturity and backfill window.",
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
            "Delay (min)": quality_assessment.observed_delay_minutes,
            "Expected lag (min)": quality_assessment.expected_delay_minutes,
            "Excess lag (min)": quality_assessment.excess_delay_minutes,
            "Status": quality_status,
            "Expected cadence / SLA": schedule_expectation_label(QUALITY_ARTIFACT_POLICY),
            "Operator meaning": quality_meaning,
            "Details": quality_result.message or (quality_result.latest_key or ""),
        }
    )

    if not freshness.empty:
        def _freshness_row(row: pd.Series) -> tuple[str, str, float, float, str]:
            if row["loader_status"] != "ok":
                return (
                    "Critical",
                    row["message"] or "Freshness could not be read.",
                    float("nan"),
                    float("nan"),
                    "",
                )
            freshness_type = str(row.get("freshness_type") or "feature")
            if freshness_type == "source":
                assessment = assess_raw_age_freshness(
                    latest_dt=row["Last updated (UTC)"],
                    policy=SOURCE_FRESHNESS_POLICY,
                    now_utc=now_utc,
                )
                return (
                    assessment.status,
                    assessment.operator_meaning,
                    float("nan"),
                    float("nan"),
                    source_expectation_label(SOURCE_FRESHNESS_POLICY),
                )
            assessment = assess_schedule_freshness(
                latest_dt=row["Last updated (UTC)"],
                policy=FEATURE_FRESHNESS_POLICY,
                now_utc=now_utc,
                upstream_latest_dt=latest_source_dt,
                upstream_label="Source freshness",
            )
            return (
                assessment.status,
                assessment.operator_meaning,
                assessment.expected_delay_minutes,
                assessment.excess_delay_minutes if assessment.excess_delay_minutes is not None else float("nan"),
                schedule_expectation_label(FEATURE_FRESHNESS_POLICY),
            )

        freshness[
            ["Status", "Operator meaning", "Expected lag (min)", "Excess lag (min)", "Expected cadence / SLA"]
        ] = freshness.apply(
            lambda row: pd.Series(_freshness_row(row)),
            axis=1,
        )
        freshness["Details"] = freshness["message"].fillna("")
        freshness = freshness.rename(columns={"source": "Data source"})
        rows.extend(
            freshness[
                [
                    "Data source",
                    "Last updated (UTC)",
                    "Delay (min)",
                    "Expected lag (min)",
                    "Excess lag (min)",
                    "Status",
                    "Expected cadence / SLA",
                    "Operator meaning",
                    "Details",
                ]
            ].to_dict("records")
        )

    return pd.DataFrame(rows)
