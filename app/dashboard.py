"""Velib Paris — Station Risk Monitor

Enterprise-grade Streamlit dashboard for bike/dock stockout prediction.
Data sources:
  • PostgreSQL (station metadata, data freshness)   — via SQLAlchemy
  • S3 Parquet (predictions, quality metrics)        — via boto3 + pandas
  • AWS CloudWatch (model & system health metrics)   — via boto3
"""
from __future__ import annotations

import os

import boto3
import streamlit as st
from sqlalchemy import create_engine, URL

from dashboard.cloudwatch import (
    build_dashboard_metric_dimensions,
    create_cloudwatch_client,
    fetch_metric_series,
)
from dashboard.queries import load_freshness, load_station_info
from dashboard.s3_loader import (
    load_latest_predictions,
    load_prediction_history,
    load_quality_recent,
)
from dashboard.targeting import resolve_dashboard_target
from dashboard.views import (
    render_alert_banner,
    render_freshness_table,
    render_history_chart,
    render_metric_section,
    render_prediction_map,
    render_status_cards,
    render_top_risk_table,
)

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Velib Paris — Station Risk Monitor",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
/* Tighten the top padding */
.block-container { padding-top: 1.2rem; }

/* Metric card polish */
[data-testid="metric-container"] {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 0.8rem 1rem;
}
[data-testid="metric-container"] label { font-size: 0.78rem; color: #6c757d; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 1.1rem; }

/* Tab bar */
[data-baseweb="tab-list"] { gap: 4px; }
[data-baseweb="tab"] { border-radius: 6px 6px 0 0; font-weight: 500; }

/* Sidebar refinement */
[data-testid="stSidebar"] { background: #f1f3f5; }
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Settings from secrets ────────────────────────────────────────────
# Supports both the documented nested format ([aws]/[app] sections) and the
# legacy flat format, so the same code works with both secrets.toml layouts.
_aws = st.secrets.get("aws", {})
_app = st.secrets.get("app", {})

AWS_PROFILE  = _aws.get("profile")      or st.secrets.get("aws_profile")
AWS_REGION   = _aws.get("region")       or st.secrets.get("region", "eu-west-3")
BUCKET       = _aws.get("bucket")       or st.secrets.get("bucket", "bikeshare-paris-387706002632-eu-west-3")
CITY         = _app.get("city")         or st.secrets.get("city", "paris")
ENVIRONMENT  = _app.get("environment")  or st.secrets.get("serving_environment", "staging")
PROJECT_SLUG = _app.get("project_slug") or st.secrets.get("project_slug", "bikeshare")
DEV_MODE     = bool(_app.get("dev_mode") if "dev_mode" in _app else st.secrets.get("dev_mode", False))
CW_NAMESPACE = _app.get("cw_custom_ns") or st.secrets.get("cw_custom_ns", "Bikeshare/Model")
THRESHOLD    = float(_app.get("threshold")    or st.secrets.get("decision_threshold", 0.37))
MODEL_VER    = _app.get("model_version") or st.secrets.get("model_version", "unknown")
PG_SCHEMA    = st.secrets.get("pg_schema", "analytics")
FRESHNESS_TABLES: list[str] = list(
    st.secrets.get(
        "freshness_tables",
        ["feat_station_snapshot_latest"],
    )
)

# Allow Docker env vars to override pg_host/pg_port so the same secrets.toml
# works both locally and inside Docker Compose on EC2.
PG_HOST = os.environ.get("STREAMLIT_PG_HOST") or str(st.secrets.get("pg_host", "localhost"))
PG_PORT = int(os.environ.get("STREAMLIT_PG_PORT") or st.secrets.get("pg_port", 15432))
PG_DB   = st.secrets.get("pg_database", "velib_dw")
PG_USER = st.secrets.get("pg_user", "velib")
PG_PASS = st.secrets.get("pg_password", "velib")

# ── Cached connections ────────────────────────────────────────────────

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


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚲 Velib Paris")
    st.markdown("---")
    target_label = st.radio(
        "Prediction target",
        ["Bike stockout", "Dock stockout"],
        index=0,
        help="Switch between predicting bike shortage vs dock shortage at each station.",
    )
    st.markdown("---")
    top_n = st.slider("Top-N stations to show", 5, 50, 20, 5)
    history_limit = st.slider("History snapshots", 12, 96, 24, 12)
    st.markdown("---")
    st.caption(f"City: {CITY}  |  Env: {ENVIRONMENT}")
    if DEV_MODE:
        st.info("DEV_MODE on — debug panels visible.")

# ── Target resolution ─────────────────────────────────────────────────
target_name = "bikes" if target_label == "Bike stockout" else "docks"
target = resolve_dashboard_target(
    target_name=target_name,
    city=CITY,
    environment=ENVIRONMENT,
    project_slug=PROJECT_SLUG,
)

# ── Page header ───────────────────────────────────────────────────────
st.markdown("## 🚲 Velib Paris — Station Risk Monitor")
render_status_cards(
    target=target,
    environment=ENVIRONMENT,
    model_version=MODEL_VER,
    threshold=THRESHOLD,
)

# ── Load data (with spinners) ─────────────────────────────────────────
with st.spinner("Loading station data…"):
    station_info = load_station_info(engine=_pg_engine(), schema=PG_SCHEMA, city=CITY)

with st.spinner("Loading latest predictions from S3…"):
    latest_predictions = load_latest_predictions(
        bucket=BUCKET, city=CITY, target_name=target_name, s3_client=_s3_client()
    )

# Alert banner (key business message — immediately visible)
render_alert_banner(predictions=latest_predictions, target=target)

# ── Tabs ──────────────────────────────────────────────────────────────
tab_map, tab_history, tab_quality, tab_system, tab_freshness = st.tabs([
    "🗺️ Live Map & Risk Table",
    "📈 Station History",
    "🔬 Prediction Quality",
    "⚙️ System Health",
    "🗂️ Data Status",
])

# ── Tab 1: Map + Risk table ───────────────────────────────────────────
with tab_map:
    selected_station = render_prediction_map(
        station_info=station_info,
        predictions=latest_predictions,
        target=target,
    )
    render_top_risk_table(predictions=latest_predictions, target=target, top_n=top_n)

# ── Tab 2: Station history ────────────────────────────────────────────
with tab_history:
    history_station = selected_station or (
        str(latest_predictions.iloc[0]["station_id"]) if not latest_predictions.empty else None
    )
    if history_station:
        with st.spinner(f"Loading history for station {history_station}…"):
            history = load_prediction_history(
                bucket=BUCKET,
                city=CITY,
                target_name=target_name,
                station_id=history_station,
                n_periods=history_limit,
                s3_client=_s3_client(),
            )
    else:
        import pandas as _pd
        history = _pd.DataFrame()

    render_history_chart(history=history, target=target, threshold=THRESHOLD)

# ── Tab 3: Prediction quality (model health) ──────────────────────────
with tab_quality:
    dims = build_dashboard_metric_dimensions(
        environment=ENVIRONMENT,
        endpoint_name=target.endpoint_name,
        city=CITY,
        target_name=target.target_name,
    )
    model_health = {
        "PR-AUC-24h": fetch_metric_series(
            _cw_client(), namespace=CW_NAMESPACE, metric_name="PR-AUC-24h", dimensions=dims
        ),
        "F1-24h": fetch_metric_series(
            _cw_client(), namespace=CW_NAMESPACE, metric_name="F1-24h", dimensions=dims
        ),
        "PredictionHeartbeat": fetch_metric_series(
            _cw_client(), namespace=CW_NAMESPACE, metric_name="PredictionHeartbeat",
            dimensions=dims, stat="Sum",
        ),
    }
    render_metric_section(title="Model performance metrics", series_map=model_health)

# ── Tab 4: System health ──────────────────────────────────────────────
with tab_system:
    sm_dims = {
        "EndpointName": target.endpoint_name,
        "VariantName": "AllTraffic",
    }
    system_health = {
        "ModelLatency": fetch_metric_series(
            _cw_client(), namespace="AWS/SageMaker", metric_name="ModelLatency",
            dimensions=sm_dims, stat="p95",
        ),
        "Invocation5XXErrors": fetch_metric_series(
            _cw_client(), namespace="AWS/SageMaker", metric_name="Invocation5XXErrors",
            dimensions=sm_dims, stat="Sum",
        ),
    }
    render_metric_section(title="Serving infrastructure health", series_map=system_health)

# ── Tab 5: Data pipeline freshness ────────────────────────────────────
with tab_freshness:
    with st.spinner("Checking data pipeline status…"):
        freshness_df = load_freshness(
            engine=_pg_engine(), schema=PG_SCHEMA, city=CITY, tables=FRESHNESS_TABLES
        )
    render_freshness_table(freshness=freshness_df)

# ── Debug (DEV_MODE only) ─────────────────────────────────────────────
if DEV_MODE:
    with st.expander("Debug: target configuration"):
        st.json({
            "target_name":      target.target_name,
            "endpoint_name":    target.endpoint_name,
            "label_column":     target.label_column,
            "score_column":     target.score_column,
            "inference_prefix": target.inference_prefix,
            "quality_prefix":   target.quality_prefix,
            "pg_host":          PG_HOST,
            "pg_port":          PG_PORT,
            "bucket":           BUCKET,
        })
