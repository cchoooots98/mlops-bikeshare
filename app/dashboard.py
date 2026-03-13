from __future__ import annotations

import time

import boto3
import pandas as pd
import streamlit as st
from pyathena import connect
from pyathena.pandas.util import as_pandas

from dashboard.cloudwatch import (
    build_dashboard_metric_dimensions,
    create_cloudwatch_client,
    fetch_metric_series,
)
from dashboard.queries import (
    build_freshness_query,
    build_latest_predictions_query,
    build_prediction_history_query,
    build_quality_summary_query,
    build_station_info_query,
)
from dashboard.targeting import resolve_dashboard_target
from dashboard.views import (
    render_freshness_table,
    render_history_chart,
    render_metric_section,
    render_prediction_map,
    render_status_cards,
    render_top_risk_table,
)


st.set_page_config(page_title="Bikeshare Business Dashboard", layout="wide")
st.title("Bikeshare Business Dashboard")

AWS_REGION = st.secrets["region"]
DATABASE = st.secrets["db"]
WORKGROUP = st.secrets["workgroup"]
ATHENA_OUTPUT = st.secrets["athena_output"]
CITY = st.secrets.get("city", "paris")
AWS_PROFILE = st.secrets.get("aws_profile", "default")
CW_NAMESPACE = st.secrets.get("cw_custom_ns", "Bikeshare/Model")
ENVIRONMENT = st.secrets.get("serving_environment", "production")
PROJECT_SLUG = st.secrets.get("project_slug", "bikeshare")
DEV_MODE = bool(st.secrets.get("dev_mode", False))
VIEW_INFO = st.secrets.get("view_station_info_latest", "v_station_information")
VIEW_PRED = st.secrets.get("view_predictions", "v_predictions")
VIEW_QUAL = st.secrets.get("view_quality", "v_quality")
FRESHNESS_TABLES = st.secrets.get(
    "freshness_tables",
    ["station_information_raw", "station_status_raw", "weather_hourly_raw", "feat_station_snapshot_latest", "inference", "monitoring_quality"],
)
CURRENT_MODEL_VERSION = st.secrets.get("model_version", "unknown")
CURRENT_THRESHOLD = float(st.secrets.get("decision_threshold", 0.37))


@st.cache_resource(show_spinner=False)
def athena_conn():
    session = boto3.Session(profile_name=AWS_PROFILE)
    return connect(
        region_name=AWS_REGION,
        s3_staging_dir=ATHENA_OUTPUT,
        work_group=WORKGROUP,
        boto3_session=session,
    )


@st.cache_data(ttl=60, show_spinner=False)
def run_athena(sql: str) -> pd.DataFrame:
    t0 = time.time()
    df = as_pandas(athena_conn().cursor().execute(sql))
    st.session_state.setdefault("_debug_last_query_ms", {})[sql[:60]] = round((time.time() - t0) * 1000, 1)
    return df


@st.cache_resource(show_spinner=False)
def cw_client():
    return create_cloudwatch_client(region_name=AWS_REGION, profile_name=AWS_PROFILE)


with st.sidebar:
    st.header("Scope")
    target_label = st.radio("Target", ["Bike stockout", "Dock stockout"], index=0)
    top_n = st.slider("Top-N risky stations", 5, 50, 20, 5)
    history_limit = st.slider("History points", 12, 96, 24, 12)
    st.caption(f"City: {CITY} | Environment: {ENVIRONMENT}")
    if DEV_MODE:
        st.info("DEV_MODE is enabled. Production-only debug publishing controls remain hidden.")

target_name = "bikes" if target_label == "Bike stockout" else "docks"
target = resolve_dashboard_target(
    target_name=target_name,
    city=CITY,
    environment=ENVIRONMENT,
    project_slug=PROJECT_SLUG,
)

render_status_cards(
    target=target,
    environment=ENVIRONMENT,
    model_version=CURRENT_MODEL_VERSION,
    threshold=CURRENT_THRESHOLD,
)

station_info = run_athena(build_station_info_query(database=DATABASE, view_name=VIEW_INFO, city=CITY))
latest_predictions = run_athena(
    build_latest_predictions_query(database=DATABASE, view_name=VIEW_PRED, city=CITY, target=target)
)
quality_rows = run_athena(
    build_quality_summary_query(database=DATABASE, view_name=VIEW_QUAL, city=CITY, target=target)
)
selected_station = render_prediction_map(station_info=station_info, predictions=latest_predictions, target=target)
render_top_risk_table(predictions=latest_predictions, target=target, top_n=top_n)

history_station = selected_station or (latest_predictions.iloc[0]["station_id"] if not latest_predictions.empty else None)
if history_station:
    history = run_athena(
        build_prediction_history_query(
            database=DATABASE,
            view_name=VIEW_PRED,
            city=CITY,
            station_id=str(history_station),
            target=target,
            limit=history_limit,
        )
    )
else:
    history = pd.DataFrame()
render_history_chart(history=history, target=target)

dims = build_dashboard_metric_dimensions(
    environment=ENVIRONMENT,
    endpoint_name=target.endpoint_name,
    city=CITY,
    target_name=target.target_name,
)
model_health = {
    "PR-AUC-24h": fetch_metric_series(cw_client(), namespace=CW_NAMESPACE, metric_name="PR-AUC-24h", dimensions=dims),
    "F1-24h": fetch_metric_series(cw_client(), namespace=CW_NAMESPACE, metric_name="F1-24h", dimensions=dims),
    "PredictionHeartbeat": fetch_metric_series(
        cw_client(), namespace=CW_NAMESPACE, metric_name="PredictionHeartbeat", dimensions=dims, stat="Sum"
    ),
}
system_health = {
    "ModelLatency": fetch_metric_series(
        cw_client(),
        namespace="AWS/SageMaker",
        metric_name="ModelLatency",
        dimensions={"EndpointName": target.endpoint_name, "VariantName": "AllTraffic"},
        stat="p95",
    ),
    "Invocation5XXErrors": fetch_metric_series(
        cw_client(),
        namespace="AWS/SageMaker",
        metric_name="Invocation5XXErrors",
        dimensions={"EndpointName": target.endpoint_name, "VariantName": "AllTraffic"},
        stat="Sum",
    ),
}
render_metric_section(title="4) Model health", series_map=model_health)
render_metric_section(title="4b) System health", series_map=system_health)

freshness_frames = [run_athena(build_freshness_query(database=DATABASE, table_name=table_name, city=CITY)) for table_name in FRESHNESS_TABLES]
render_freshness_table(freshness=pd.concat(freshness_frames, ignore_index=True) if freshness_frames else pd.DataFrame())

with st.expander("Debug: query timings (ms)"):
    st.write(st.session_state.get("_debug_last_query_ms", {}))

with st.expander("Debug: target configuration"):
    st.json(
        {
            "target_name": target.target_name,
            "endpoint_name": target.endpoint_name,
            "label_column": target.label_column,
            "score_column": target.score_column,
            "inference_prefix": target.inference_prefix,
            "quality_prefix": target.quality_prefix,
            "quality_row_count": int(len(quality_rows)),
        }
    )
