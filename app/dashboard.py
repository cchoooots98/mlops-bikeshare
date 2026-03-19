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
[data-baseweb="tab-list"] { gap: 4px; }
[data-baseweb="tab"] { border-radius: 6px 6px 0 0; font-weight: 500; }
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
    st.caption(f"City: {CITY} | Env: {ENVIRONMENT}")
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
    station_info = load_station_info(engine=_pg_engine(), schema=PG_SCHEMA, city=CITY)

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
    freshness_result = load_freshness(
        engine=_pg_engine(),
        schema=PG_SCHEMA,
        city=CITY,
        tables=FRESHNESS_TABLES,
    )

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

render_alert_banner(
    prediction_result=latest_predictions,
    target=target,
    threshold=threshold,
    stale_after_minutes=PREDICTION_STALE_AFTER_MINUTES,
)

tab_map, tab_history, tab_quality, tab_system, tab_status = st.tabs(
    [
        "Live Map & Risk Table",
        "Station History",
        "Prediction Quality",
        "System Health",
        "Data Status",
    ]
)

with tab_map:
    selected_station = render_prediction_map(
        station_info=station_info,
        prediction_result=latest_predictions,
        target=target,
        threshold=threshold,
    )
    render_top_risk_table(
        prediction_result=latest_predictions,
        target=target,
        top_n=top_n,
        threshold=threshold,
    )

with tab_history:
    history_station = selected_station or (
        str(latest_predictions.data.iloc[0]["station_id"]) if latest_predictions.ok and not latest_predictions.data.empty else None
    )
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
            message="Select a station on the map after prediction data becomes available.",
            source_name="Prediction history",
        )
    render_history_chart(history_result=history_result, target=target, threshold=threshold)

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
    }
    render_metric_section(title="Model performance metrics", series_map=model_health)

with tab_system:
    sm_dims = {
        "EndpointName": target.endpoint_name,
        "VariantName": "AllTraffic",
    }
    system_health = {
        "ModelLatency": fetch_metric_series(
            _cw_client(),
            namespace="AWS/SageMaker",
            metric_name="ModelLatency",
            dimensions=sm_dims,
            stat="p95",
        ),
        "Invocation5XXErrors": fetch_metric_series(
            _cw_client(),
            namespace="AWS/SageMaker",
            metric_name="Invocation5XXErrors",
            dimensions=sm_dims,
            stat="Sum",
        ),
    }
    render_metric_section(title="Serving infrastructure health", series_map=system_health)

with tab_status:
    render_data_status_table(
        prediction_result=latest_predictions,
        quality_result=latest_quality,
        freshness_result=freshness_result,
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
