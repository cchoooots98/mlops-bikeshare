"""Velib Paris Station Risk Monitor."""
from __future__ import annotations

import os
from datetime import datetime, timezone

import boto3
import pandas as pd
import streamlit as st
from sqlalchemy import URL, create_engine

from dashboard.cloudwatch import (
    build_dashboard_metric_dimensions,
    create_cloudwatch_client,
    fetch_metric_series,
)
from dashboard.contracts import ArtifactLoadResult, FreshnessLoadResult, LoadStatus
from dashboard.metadata import load_dashboard_model_metadata
from dashboard.presentation import MetricSpec, build_station_risk_frame, resolve_selected_station
from dashboard.queries import load_freshness, load_station_info
from dashboard.s3_loader import (
    load_latest_predictions,
    load_latest_quality_status,
    load_prediction_history,
)
from dashboard.targeting import resolve_dashboard_target
from dashboard.views import (
    render_alert_banner,
    render_data_status_table,
    render_history_chart,
    render_metric_section,
    render_prediction_map,
    render_quality_status_panel,
    render_selected_station_summary,
    render_status_cards,
    render_top_risk_table,
)

st.set_page_config(
    page_title="Velib Paris Station Risk Monitor",
    page_icon=":bike:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; }
[data-testid="metric-container"] {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 0.8rem 1rem;
}
[data-testid="metric-container"] label { font-size: 0.78rem; color: #6c757d; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 1.1rem; }
[data-baseweb="tab-list"] { gap: 18px; }
[data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.7rem 1rem 0.9rem;
}
[data-testid="stSidebar"] { background: #f1f3f5; }
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

_aws = st.secrets.get("aws", {})
_app = st.secrets.get("app", {})

AWS_PROFILE = _aws.get("profile") or st.secrets.get("aws_profile")
AWS_REGION = _aws.get("region") or st.secrets.get("region", "eu-west-3")
BUCKET = _aws.get("bucket") or st.secrets.get("bucket", "bikeshare-paris-387706002632-eu-west-3")
CITY = _app.get("city") or st.secrets.get("city", "paris")
ENVIRONMENT = _app.get("environment") or st.secrets.get("serving_environment", "staging")
PROJECT_SLUG = _app.get("project_slug") or st.secrets.get("project_slug", "bikeshare")
DEV_MODE = bool(_app.get("dev_mode") if "dev_mode" in _app else st.secrets.get("dev_mode", False))
CW_NAMESPACE = _app.get("cw_custom_ns") or st.secrets.get("cw_custom_ns", "Bikeshare/Model")
DEFAULT_THRESHOLD = float(_app.get("threshold") or st.secrets.get("decision_threshold", 0.37))
MODEL_VER = _app.get("model_version") or st.secrets.get("model_version", "unknown")
PG_SCHEMA = st.secrets.get("pg_schema", "analytics")
PREDICTION_STALE_AFTER_MINUTES = int(
    _app.get("prediction_stale_after_minutes")
    or st.secrets.get("prediction_stale_after_minutes", 30)
)
QUALITY_STALE_AFTER_MINUTES = int(
    _app.get("quality_stale_after_minutes")
    or st.secrets.get("quality_stale_after_minutes", 45)
)
FEATURE_STALE_AFTER_MINUTES = int(
    _app.get("feature_stale_after_minutes")
    or st.secrets.get("feature_stale_after_minutes", 60)
)
FRESHNESS_TABLES: list[str] = list(
    st.secrets.get(
        "freshness_tables",
        ["feat_station_snapshot_latest"],
    )
)

PG_HOST = os.environ.get("STREAMLIT_PG_HOST") or str(st.secrets.get("pg_host", "localhost"))
PG_PORT = int(os.environ.get("STREAMLIT_PG_PORT") or st.secrets.get("pg_port", 15432))
PG_DB = st.secrets.get("pg_database", "velib_dw")
PG_USER = st.secrets.get("pg_user", "velib")
PG_PASS = st.secrets.get("pg_password", "velib")


@st.cache_resource(show_spinner=False)
def _pg_engine():
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=PG_USER,
        password=PG_PASS,
        host=PG_HOST,
        port=PG_PORT,
        database=PG_DB,
    )
    return create_engine(url, pool_pre_ping=True)


@st.cache_resource(show_spinner=False)
def _boto_session():
    if AWS_PROFILE:
        return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    return boto3.Session(region_name=AWS_REGION)


@st.cache_resource(show_spinner=False)
def _s3_client():
    return _boto_session().client("s3")


@st.cache_resource(show_spinner=False)
def _cw_client():
    return create_cloudwatch_client(region_name=AWS_REGION, profile_name=AWS_PROFILE)


def _determine_pipeline_state(
    *,
    prediction_result: ArtifactLoadResult,
    freshness_result: FreshnessLoadResult,
    stale_after_minutes: int,
) -> str:
    if freshness_result.status != LoadStatus.OK:
        return "Degraded"
    if prediction_result.status in {
        LoadStatus.ALL_SCORES_NULL,
        LoadStatus.ACCESS_DENIED,
        LoadStatus.READ_ERROR,
        LoadStatus.SCHEMA_ERROR,
    }:
        return "Failed"
    if prediction_result.status == LoadStatus.NO_OBJECTS:
        return "Degraded"
    if prediction_result.latest_dt is not None:
        age_minutes = (datetime.now(timezone.utc) - prediction_result.latest_dt).total_seconds() / 60
        if age_minutes > stale_after_minutes:
            return "Degraded"
    freshness = freshness_result.data
    if not freshness.empty and (
        (freshness["loader_status"] != "ok").any()
        or pd.to_datetime(
            freshness["latest_dt_str"], format="%Y-%m-%d-%H-%M", errors="coerce", utc=True
        )
        .pipe(lambda series: ((datetime.now(timezone.utc) - series).dt.total_seconds() / 60 >= 60).any())
    ):
        return "Degraded"
    return "Healthy"


def _load_station_info_safely() -> tuple[pd.DataFrame, str]:
    try:
        return load_station_info(engine=_pg_engine(), schema=PG_SCHEMA, city=CITY), ""
    except Exception as exc:
        return pd.DataFrame(), f"Station metadata is unavailable because PostgreSQL could not be reached: {exc}"


def _load_freshness_safely() -> FreshnessLoadResult:
    try:
        return load_freshness(
            engine=_pg_engine(),
            schema=PG_SCHEMA,
            city=CITY,
            tables=FRESHNESS_TABLES,
        )
    except Exception as exc:
        return FreshnessLoadResult(
            status=LoadStatus.READ_ERROR,
            data=pd.DataFrame(),
            message=f"Feature freshness is unavailable because PostgreSQL could not be reached: {exc}",
        )


def _model_latency_ms(series: pd.DataFrame) -> pd.DataFrame:
    if series.empty or "ModelLatency" not in series.columns:
        return series
    converted = series.copy()
    converted["ModelLatency"] = pd.to_numeric(converted["ModelLatency"], errors="coerce") / 1000.0
    return converted


with st.sidebar:
    st.markdown("### Velib Paris")
    st.markdown("---")
    target_label = st.radio(
        "Prediction target",
        ["Bike stockout", "Dock stockout"],
        index=0,
        help="Switch between predicting bike shortage versus dock shortage at each station.",
    )
    st.markdown("---")
    top_n = st.slider("Top-N stations to show", 5, 50, 20, 5)
    history_limit = st.slider("History snapshots", 12, 96, 24, 12)
    st.markdown("---")
    if DEV_MODE:
        st.info("Development mode is enabled. Operator debug details are visible.")

target_name = "bikes" if target_label == "Bike stockout" else "docks"
target = resolve_dashboard_target(
    target_name=target_name,
    city=CITY,
    environment=ENVIRONMENT,
    project_slug=PROJECT_SLUG,
)
model_metadata = load_dashboard_model_metadata(
    target_name=target_name,
    environment=ENVIRONMENT,
    fallback_model_version=MODEL_VER,
)
threshold = model_metadata.threshold if model_metadata.threshold is not None else DEFAULT_THRESHOLD

st.markdown("## Velib Paris Station Risk Monitor")

with st.spinner("Loading station metadata..."):
    station_info, station_info_error = _load_station_info_safely()

with st.spinner("Loading latest prediction artifact..."):
    latest_predictions = load_latest_predictions(
        bucket=BUCKET,
        city=CITY,
        target_name=target_name,
        s3_client=_s3_client(),
    )

with st.spinner("Loading latest quality artifact..."):
    latest_quality = load_latest_quality_status(
        bucket=BUCKET,
        city=CITY,
        target_name=target_name,
        s3_client=_s3_client(),
    )

with st.spinner("Checking feature freshness..."):
    freshness_result = _load_freshness_safely()

pipeline_state = _determine_pipeline_state(
    prediction_result=latest_predictions,
    freshness_result=freshness_result,
    stale_after_minutes=PREDICTION_STALE_AFTER_MINUTES,
)

render_status_cards(
    target=target,
    environment=ENVIRONMENT,
    pipeline_state=pipeline_state,
    endpoint_name=target.endpoint_name,
    model_version=model_metadata.display_version,
    latest_prediction_dt=latest_predictions.latest_dt,
    threshold=threshold,
)
if model_metadata.message:
    st.warning(model_metadata.message)
if station_info_error:
    st.warning(station_info_error)
if freshness_result.message and freshness_result.status != LoadStatus.OK:
    st.warning(freshness_result.message)

render_alert_banner(
    prediction_result=latest_predictions,
    target=target,
    threshold=threshold,
    stale_after_minutes=PREDICTION_STALE_AFTER_MINUTES,
)

station_risk_frame = build_station_risk_frame(
    station_info=station_info,
    prediction_result=latest_predictions,
    target_name=target.target_name,
    threshold=threshold,
)

tab_map, tab_history, tab_quality, tab_system, tab_status = st.tabs(
    [
        "Live Ops",
        "Station History",
        "Prediction Quality",
        "System Health",
        "Data Status",
    ]
)

with tab_map:
    render_prediction_map(
        station_risk_frame=station_risk_frame,
        target=target,
        threshold=threshold,
    )
    selected_station = resolve_selected_station(
        station_risk_frame=station_risk_frame,
        selected_station_id=st.session_state.get("selected_station_id"),
    )
    if selected_station is not None:
        st.session_state["selected_station_id"] = str(selected_station["station_id"])
        st.session_state["selected_station_name"] = str(selected_station["station_name"])
    render_selected_station_summary(selected_station=selected_station, target=target)
    render_top_risk_table(
        station_risk_frame=station_risk_frame,
        top_n=top_n,
        target=target,
    )

with tab_history:
    selected_station = resolve_selected_station(
        station_risk_frame=station_risk_frame,
        selected_station_id=st.session_state.get("selected_station_id"),
    )
    history_station = str(selected_station["station_id"]) if selected_station is not None else None
    if history_station:
        with st.spinner(f"Loading history for station {history_station}..."):
            history_result = load_prediction_history(
                bucket=BUCKET,
                city=CITY,
                target_name=target_name,
                station_id=history_station,
                n_periods=history_limit,
                s3_client=_s3_client(),
            )
    else:
        history_result = ArtifactLoadResult(
            status=LoadStatus.NO_OBJECTS,
            message="A station will appear here as soon as prediction data becomes available.",
            source_name="Prediction history",
        )
    render_history_chart(
        history_result=history_result,
        target=target,
        threshold=threshold,
        selected_station=selected_station,
    )

with tab_quality:
    dims = build_dashboard_metric_dimensions(
        environment=ENVIRONMENT,
        endpoint_name=target.endpoint_name,
        city=CITY,
        target_name=target.target_name,
    )
    model_health = {
        "PR-AUC-24h": fetch_metric_series(
            _cw_client(),
            namespace=CW_NAMESPACE,
            metric_name="PR-AUC-24h",
            dimensions=dims,
        ),
        "F1-24h": fetch_metric_series(
            _cw_client(),
            namespace=CW_NAMESPACE,
            metric_name="F1-24h",
            dimensions=dims,
        ),
        "PredictionHeartbeat": fetch_metric_series(
            _cw_client(),
            namespace=CW_NAMESPACE,
            metric_name="PredictionHeartbeat",
            dimensions=dims,
            stat="Sum",
        ),
        "ThresholdHitRate-24h": fetch_metric_series(
            _cw_client(),
            namespace=CW_NAMESPACE,
            metric_name="ThresholdHitRate-24h",
            dimensions=dims,
        ),
        "Samples-24h": fetch_metric_series(
            _cw_client(),
            namespace=CW_NAMESPACE,
            metric_name="Samples-24h",
            dimensions=dims,
            stat="Sum",
        ),
    }
    render_quality_status_panel(
        quality_result=latest_quality,
        metric_series_map=model_health,
    )
    render_metric_section(
        title="Prediction Quality",
        description="Last 24 hours in UTC. These metrics summarize mature quality evidence and published monitoring signals.",
        series_map=model_health,
        quality_result=latest_quality,
        metric_specs={
            "PR-AUC-24h": MetricSpec(label="PR-AUC (24h)", direction="higher", warning=0.70, critical=0.55, decimals=3),
            "F1-24h": MetricSpec(label="F1 (24h)", direction="higher", warning=0.55, critical=0.40, decimals=3),
            "PredictionHeartbeat": MetricSpec(label="Prediction Heartbeat (24h)", direction="higher", warning=1.0, critical=1.0, decimals=0),
            "ThresholdHitRate-24h": MetricSpec(label="Threshold Hit Rate (24h)", direction="none", decimals=3),
            "Samples-24h": MetricSpec(label="Samples (24h)", direction="higher", warning=1.0, critical=1.0, decimals=0),
        },
        key_prefix="prediction-quality",
    )

with tab_system:
    dims = build_dashboard_metric_dimensions(
        environment=ENVIRONMENT,
        endpoint_name=target.endpoint_name,
        city=CITY,
        target_name=target.target_name,
    )
    sm_dims = {
        "EndpointName": target.endpoint_name,
        "VariantName": "AllTraffic",
    }
    system_health = {
        "ModelLatency": _model_latency_ms(
            fetch_metric_series(
                _cw_client(),
                namespace="AWS/SageMaker",
                metric_name="ModelLatency",
                dimensions=sm_dims,
                stat="p95",
            )
        ),
        "Invocation5XXErrors": fetch_metric_series(
            _cw_client(),
            namespace="AWS/SageMaker",
            metric_name="Invocation5XXErrors",
            dimensions=sm_dims,
            stat="Sum",
        ),
        "Invocation4XXErrors": fetch_metric_series(
            _cw_client(),
            namespace="AWS/SageMaker",
            metric_name="Invocation4XXErrors",
            dimensions=sm_dims,
            stat="Sum",
        ),
        "Invocations": fetch_metric_series(
            _cw_client(),
            namespace="AWS/SageMaker",
            metric_name="Invocations",
            dimensions=sm_dims,
            stat="Sum",
        ),
        "PredictionHeartbeat": fetch_metric_series(
            _cw_client(),
            namespace=CW_NAMESPACE,
            metric_name="PredictionHeartbeat",
            dimensions=dims,
            stat="Sum",
        ),
        "PSI": fetch_metric_series(
            _cw_client(),
            namespace=CW_NAMESPACE,
            metric_name="PSI",
            dimensions=dims,
        ),
    }
    render_metric_section(
        title="System Health",
        description="Serving SLA view for the last 24 hours in UTC.",
        series_map=system_health,
        metric_specs={
            "ModelLatency": MetricSpec(
                label="ModelLatency p95 (ms)",
                direction="lower",
                warning=200.0,
                critical=300.0,
                decimals=0,
                empty_message="No SageMaker latency samples are available for the selected endpoint.",
            ),
            "Invocation5XXErrors": MetricSpec(
                label="Invocation5XXErrors (24h)",
                direction="lower",
                warning=0.0,
                critical=0.0,
                decimals=0,
                empty_message="No 5xx error samples are available for the selected endpoint.",
            ),
            "Invocation4XXErrors": MetricSpec(
                label="Invocation4XXErrors (24h)",
                direction="lower",
                warning=0.0,
                critical=10.0,
                decimals=0,
                empty_message="No 4xx error samples are available for the selected endpoint.",
            ),
            "Invocations": MetricSpec(
                label="Invocations (24h)",
                direction="higher",
                warning=1.0,
                critical=1.0,
                decimals=0,
                empty_message="No invocation count is available for the selected endpoint.",
            ),
            "PredictionHeartbeat": MetricSpec(
                label="Prediction Heartbeat (24h)",
                direction="higher",
                warning=1.0,
                critical=1.0,
                decimals=0,
                empty_message="Heartbeat samples are missing for the selected target and environment.",
            ),
            "PSI": MetricSpec(
                label="PSI (24h)",
                direction="lower",
                warning=0.20,
                critical=0.30,
                decimals=3,
                empty_message="No PSI drift samples are available yet for the selected target.",
            ),
        },
        key_prefix="system-health",
    )
    st.caption(
        "Runbook thresholds: latency warning 200 ms / critical 300 ms, 5xx must remain at 0, "
        "and PSI warning/critical bands are 0.20 / 0.30."
    )

with tab_status:
    render_data_status_table(
        prediction_result=latest_predictions,
        quality_result=latest_quality,
        freshness_result=freshness_result,
        prediction_sla_minutes=PREDICTION_STALE_AFTER_MINUTES,
        quality_sla_minutes=QUALITY_STALE_AFTER_MINUTES,
        feature_sla_minutes=FEATURE_STALE_AFTER_MINUTES,
    )

if DEV_MODE:
    with st.expander("Debug target configuration"):
        st.json(
            {
                "target_name": target.target_name,
                "endpoint_name": target.endpoint_name,
                "prediction_status": latest_predictions.status.value,
                "prediction_key": latest_predictions.latest_key,
                "quality_status": latest_quality.status.value,
                "quality_key": latest_quality.latest_key,
                "bucket": BUCKET,
                "pg_host": PG_HOST,
                "pg_port": PG_PORT,
            }
        )
