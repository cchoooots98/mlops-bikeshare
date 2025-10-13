# app/dashboard.py
# -*- coding: utf-8 -*-
"""
Bikeshare Business Dashboard (T+30 task)
- Map: color stations by P(stockout @ t+30). Sidebar shows station info + last-2h risk trajectory (yhat_bikes history).
- Top-N risky stations: filter by region/radius; show suggested rebalancing quantity (simple heuristic).
- Model health: PR-AUC/F1/threshold hit-rate, drift metrics (CloudWatch custom metrics + Athena if available).
- System health: endpoint latency/error rate/batch success (CloudWatch native + custom heartbeat).
- Data freshness: latest partitions across raw/features/inference/quality and failed-batch counters.
- Data sources: Athena (tables/views) + CloudWatch metrics (GetMetricData).
"""

import math
import time
from datetime import datetime, timedelta, timezone

import streamlit as st
import pandas as pd
from pyathena import connect
from pyathena.pandas.util import as_pandas
import boto3
from streamlit_folium import st_folium
import folium
import plotly.express as px

# ------------------------------
# 0) App-level config and caching
# ------------------------------
st.set_page_config(page_title="Bikeshare Ops – Business Dashboard", layout="wide")
st.title("🚲 Bikeshare – Business Dashboard (T+30)")

# Read config from .streamlit/secrets.toml to avoid hardcoding
REGION = st.secrets["region"]
DB = st.secrets["db"]
WG = st.secrets["workgroup"]
ATH_OUT = st.secrets["athena_output"]
CITY = st.secrets.get("city", "nyc")
AWS_PROFILE = st.secrets.get("aws_profile", "default")

CW_NS = st.secrets.get("cw_custom_ns", "Bikeshare/Model")
SM_ENDPOINT = st.secrets.get("sm_endpoint", "bikeshare-prod")

# View / table names (can be overridden in secrets.toml)
VIEW_INFO = st.secrets.get("view_station_info_latest", "v_station_information")
VIEW_PRED = st.secrets.get("view_predictions", "v_predictions")
VIEW_QUAL = st.secrets.get("view_quality", "v_quality")

TBL_INFO     = st.secrets.get("tbl_station_info_raw", "station_information_raw")
TBL_STATUS   = st.secrets.get("tbl_station_status_raw", "station_status_raw")
TBL_WEATHER  = st.secrets.get("tbl_weather_hourly_raw", "weather_hourly_raw")
TBL_FEATURES = st.secrets.get("tbl_features", "features_offline")
TBL_INFER    = st.secrets.get("tbl_inference", "inference")
TBL_MON_QUAL = st.secrets.get("tbl_monitoring_quality", "monitoring_quality")

# Use Streamlit cache to keep page load < 3s while remaining fresh
@st.cache_resource(show_spinner=False)
def athena_conn():
    """Create an Athena connection (cached)."""
    # Create a session with the specified profile
    session = boto3.Session(profile_name=AWS_PROFILE)
    return connect(
        region_name=REGION,
        s3_staging_dir=ATH_OUT,
        work_group=WG,
        boto3_session=session,
    )

@st.cache_data(ttl=60, show_spinner=False)
def run_athena(sql: str) -> pd.DataFrame:
    """Run an Athena query and return a pandas DataFrame (cached for 60s)."""
    t0 = time.time()
    df = as_pandas(athena_conn().cursor().execute(sql))
    st.session_state.setdefault("_debug_last_query_ms", {})[sql[:60]] = round((time.time() - t0) * 1000, 1)
    return df

@st.cache_resource(show_spinner=False)
def cw_client():
    """Create a CloudWatch client (cached)."""
    session = boto3.Session(profile_name=AWS_PROFILE)
    return session.client("cloudwatch", region_name=REGION)

def list_available_metrics(namespace: str) -> pd.DataFrame:
    """List all available metrics in a namespace for debugging."""
    try:
        cw = cw_client()
        paginator = cw.get_paginator('list_metrics')
        metrics = []
        
        for page in paginator.paginate(Namespace=namespace):
            for metric in page['Metrics']:
                metrics.append({
                    'MetricName': metric['MetricName'],
                    'Dimensions': {d['Name']: d['Value'] for d in metric.get('Dimensions', [])}
                })
        
        return pd.DataFrame(metrics)
    except Exception as e:
        st.error(f"Error listing metrics for {namespace}: {str(e)}")
        return pd.DataFrame()

def generate_mock_metric_data(metric_name: str, hours: int = 24) -> pd.DataFrame:
    """Generate realistic mock CloudWatch metric data for demonstration."""
    import numpy as np
    
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours)
    
    # Generate timestamps every 5 minutes
    timestamps = pd.date_range(start=start_time, end=end_time, freq='5min')
    
    # Generate realistic values based on metric type
    np.random.seed(42)  # For reproducible demo data
    
    if metric_name == "PR-AUC-24h":
        # PR-AUC should be between 0.7-0.95, with some variation
        values = 0.85 + 0.05 * np.sin(np.arange(len(timestamps)) * 0.1) + 0.02 * np.random.normal(0, 1, len(timestamps))
        values = np.clip(values, 0.7, 0.95)
    elif metric_name == "F1-24h":
        # F1 score should be between 0.6-0.9
        values = 0.75 + 0.08 * np.sin(np.arange(len(timestamps)) * 0.15) + 0.03 * np.random.normal(0, 1, len(timestamps))
        values = np.clip(values, 0.6, 0.9)
    elif metric_name == "ThresholdHitRate-24h":
        # Hit rate should be between 0.8-1.0
        values = 0.92 + 0.05 * np.sin(np.arange(len(timestamps)) * 0.2) + 0.02 * np.random.normal(0, 1, len(timestamps))
        values = np.clip(values, 0.8, 1.0)
    elif metric_name == "Samples-24h":
        # Number of samples should be between 1000-5000
        base = 2500 + 1000 * np.sin(np.arange(len(timestamps)) * 0.1)  # Daily pattern
        values = base + 200 * np.random.normal(0, 1, len(timestamps))
        values = np.clip(values, 1000, 5000)
    elif metric_name == "ModelLatency":
        # Model latency in ms, should be 50-200ms
        values = 100 + 20 * np.sin(np.arange(len(timestamps)) * 0.05) + 10 * np.random.normal(0, 1, len(timestamps))
        values = np.clip(values, 50, 200)
    elif metric_name == "OverheadLatency":
        # Overhead latency, should be 10-50ms
        values = 25 + 8 * np.sin(np.arange(len(timestamps)) * 0.03) + 3 * np.random.normal(0, 1, len(timestamps))
        values = np.clip(values, 10, 50)
    elif "Error" in metric_name:
        # Error rates should be low, 0-10 errors per 5min
        values = 2 + np.random.poisson(1, len(timestamps))
        values = np.clip(values, 0, 10)
    elif metric_name == "PredictionHeartbeat":
        # Heartbeat should be mostly 1 (success), occasionally 0
        values = np.random.choice([0, 1], len(timestamps), p=[0.05, 0.95])
    else:
        # Default: random values between 0 and 100
        values = 50 + 20 * np.sin(np.arange(len(timestamps)) * 0.1) + 10 * np.random.normal(0, 1, len(timestamps))
        values = np.clip(values, 0, 100)
    
    return pd.DataFrame({
        "ts": timestamps,
        metric_name: values
    })

def check_endpoint_status():
    """Check if SageMaker endpoint exists and is in service."""
    try:
        session = boto3.Session(profile_name=AWS_PROFILE)
        sagemaker = session.client('sagemaker', region_name=REGION)
        
        response = sagemaker.describe_endpoint(EndpointName=SM_ENDPOINT)
        status = response['EndpointStatus']
        
        if status == 'InService':
            st.success(f"✅ SageMaker endpoint '{SM_ENDPOINT}' is InService")
            return True
        else:
            st.warning(f"⚠️ SageMaker endpoint '{SM_ENDPOINT}' status: {status}")
            return False
    except Exception as e:
        st.error(f"❌ Cannot find SageMaker endpoint '{SM_ENDPOINT}': {str(e)}")
        return False

# ------------------------------
# 1) Sidebar: global filters
# ------------------------------
with st.sidebar:
    st.header("Filters")
    # Number of hours of *history* to plot for the selected station's T+30 risk trajectory
    history_hours = st.slider("History window (hours) for T+30 risk", 1, 6, 2, 1)
    top_n = st.slider("Top-N risky stations", 5, 50, 20, 5)
    radius_km = st.slider("Radius filter (km)", 0, 5, 0, 1)
    st.caption(f"City: **{CITY}** • Region: **{REGION}** • Endpoint: **{SM_ENDPOINT}**")
    
    st.info("📊 **Real CloudWatch Mode**: Querying live metrics from AWS CloudWatch")
    
    with st.expander("📋 Required CloudWatch Metrics"):
        st.markdown("""
        **Custom Metrics (Namespace: Bikeshare/Model):**
        - PR-AUC-24h
        - F1-24h  
        - ThresholdHitRate-24h
        - Samples-24h
        - PredictionHeartbeat
        - PSI
        
        **SageMaker Metrics (Auto-generated):**
        - ModelLatency
        - OverheadLatency
        - Invocation4XXErrors
        - Invocation5XXErrors
        """)

# Set demo_mode to False for real CloudWatch data only
demo_mode = False

# ------------------------------
# 2) Athena data pulls
# ------------------------------
# 2.1 Latest station info (lat/lon/capacity) – one row per station
sql_info = f"""
WITH t AS (
  SELECT
    station_id, name, capacity, lat, lon, dt_ts,
    row_number() OVER (PARTITION BY station_id ORDER BY dt_ts DESC) AS rn
  FROM {DB}.{VIEW_INFO}
  WHERE city = '{CITY}'
)
SELECT station_id, name, capacity, lat, lon
FROM t
WHERE rn = 1
"""
df_info = run_athena(sql_info)

# 2.2 Latest T+30 risk per station - use UTC timezone for proper comparison
sql_pred_latest = f"""
WITH p AS (
    SELECT
        CAST(station_id AS varchar) AS station_id, 
        LOWER(CAST(station_id AS varchar)) AS station_id_norm,
        TRY(date_parse(dt, '%%Y-%%m-%%d-%%H-%%i')) AS ts,
        CAST(yhat_bikes AS double) AS yhat_bikes,
        row_number() OVER (
            PARTITION BY LOWER(CAST(station_id AS varchar))
            ORDER BY TRY(date_parse(dt, '%%Y-%%m-%%d-%%H-%%i')) DESC
        ) AS rn
    FROM {DB}.{VIEW_PRED}
    WHERE city = '{CITY}'
        AND yhat_bikes IS NOT NULL
)
SELECT station_id, station_id_norm, ts, yhat_bikes
FROM p
WHERE rn = 1
"""
df_pred_latest = run_athena(sql_pred_latest)

# 2.3 Time series for the selected station (use UTC timezone)
def sql_pred_series(station_id: str) -> str:
    """Build SQL for T+30 risk history for a single station - show latest N predictions regardless of timestamp."""
    return f"""
    SELECT
      CAST(station_id AS varchar) AS station_id,
      TRY(date_parse(dt, '%Y-%m-%d-%H-%i')) AS ts,
      CAST(yhat_bikes AS double) AS yhat_bikes
    FROM {DB}.{VIEW_PRED}
    WHERE city = '{CITY}'
      AND CAST(station_id AS varchar) = '{station_id}'
      AND yhat_bikes IS NOT NULL
      AND TRY(date_parse(dt, '%Y-%m-%d-%H-%i')) IS NOT NULL
    ORDER BY TRY(date_parse(dt, '%Y-%m-%d-%H-%i')) DESC
    LIMIT {history_hours * 6}
    """

# 2.4 Quality join (for AUC/F1/threshold context / 24h trend) - use UTC timezone
sql_quality = f"""
SELECT
  TRY(date_parse(dt, '%%Y-%%m-%%d-%%H-%%i')) AS ts_pred,
  TRY(date_parse(dt_plus30, '%%Y-%%m-%%d-%%H-%%i')) AS ts_plus30,
  station_id,
  CAST(yhat_bikes AS double) AS yhat_bikes,
  CAST(y_stockout_bikes_30 AS tinyint) AS y_true_30
FROM {DB}.{VIEW_QUAL}
WHERE city = '{CITY}'
  AND TRY(date_parse(dt, '%%Y-%%m-%%d-%%H-%%i')) >= current_timestamp AT TIME ZONE 'UTC' - INTERVAL '24' hour
"""
df_quality = run_athena(sql_quality)

# ------------------------------
# 3) City Map (Folium) – color by yhat_bikes (T+30 risk)
# ------------------------------
st.subheader("1) City Map — Risk heat by P(stockout @ t+30) + last-2h trajectory")

# Build mapping DataFrame: station meta + latest risk
if not df_info.empty:
    df_info["station_id"] = df_info["station_id"].astype(str).str.strip()
    if not df_pred_latest.empty:
        df_pred_latest["station_id"] = df_pred_latest["station_id"].astype(str).str.strip()
        df_pred_latest["yhat_bikes"] = pd.to_numeric(df_pred_latest["yhat_bikes"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    
    # Merge predictions onto station info
    if not df_pred_latest.empty:
        df_map = df_info.merge(
            df_pred_latest[["station_id", "yhat_bikes"]].rename(columns={"yhat_bikes": "risk"}),
            on="station_id", how="left"
        ).fillna({"risk": 0.0})
    else:
        df_map = df_info.assign(risk=0.0)
else:
    df_map = pd.DataFrame()

# Render Folium map
if df_map.empty:
    st.warning("No station data found. Check Athena views or time filters.")
    map_state = {}
else:
    center_lat, center_lon = df_map["lat"].mean(), df_map["lon"].mean()
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12, control_scale=True)

    # Color stations by risk: green -> orange -> red (more sensitive thresholds)
    def risk_color(p: float) -> str:
        # p in [0, 1]; more sensitive bands for better visualization
        if p >= 0.15:      # 15% or higher = red (high risk)
            return "red"
        elif p >= 0.01:   # 1% to 15% = orange (medium risk)
            return "orange"
        else:             # Below 1% = green (low risk)
            return "green"

    for _, row in df_map.iterrows():
        cap_txt = int(row["capacity"]) if pd.notnull(row["capacity"]) else "NA"
        risk_txt = f"{row['risk']:.4f}" if row["risk"] < 0.01 else f"{row['risk']:.2f}"
        popup_html = (
            f"<b>{row['name']}</b><br/>"
            f"Station ID: {row['station_id']}<br/>"
            f"Capacity: {cap_txt}<br/>"
            f"P@t+30: {risk_txt}"
        )
        popup = folium.Popup(popup_html, max_width=300)
        folium.CircleMarker(
            radius=5,
            location=[row["lat"], row["lon"]],
            color=risk_color(row["risk"]),
            fill=True,
            fill_color=risk_color(row["risk"]),
            fill_opacity=0.7,
        ).add_child(popup).add_to(fmap)

    map_state = st_folium(fmap, width=900, height=520)

# Station selection: pick nearest station to last click (if any); else default top-1 risky
selected_station_id = None
if map_state and map_state.get("last_clicked"):
    click_lat = map_state["last_clicked"]["lat"]
    click_lon = map_state["last_clicked"]["lng"]

    # Haversine distance helper
    def hav(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
        return 2 * R * math.asin(math.sqrt(a))

    df_map["km"] = df_map.apply(lambda r: hav(click_lat, click_lon, r["lat"], r["lon"]), axis=1)
    selected_station_id = df_map.loc[df_map["km"].idxmin(), "station_id"]
elif not df_map.empty:
    # Default to the riskiest station on the map
    selected_station_id = df_map.sort_values("risk", ascending=False).iloc[0]["station_id"]


# Plot last-N-hours risk trajectory for the selected station
if selected_station_id:
    st.caption(f"Selected station ID: **{selected_station_id}**")
    series = run_athena(sql_pred_series(selected_station_id))
    if not series.empty:
        # Sort by timestamp ascending for proper chronological plotting
        series = series.sort_values("ts")
        fig = px.line(series, x="ts", y="yhat_bikes",
                      labels={"yhat_bikes": "P(stockout @ t+30)", "ts": "Timestamp"},
                      title=f"Latest {len(series)} predictions — P(stockout) for T+30")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No prediction rows found for this station.")

st.markdown("---")

# ------------------------------
# 4) Top-N risk stations (with region/radius filter)
# ------------------------------
st.subheader("2) Top-N Risk Stations — optional radius & suggested rebalancing")


df_top = df_map.copy()
if radius_km and map_state and map_state.get("last_clicked"):
    lat0 = map_state["last_clicked"]["lat"]
    lon0 = map_state["last_clicked"]["lng"]
    def hav2(r):
        R = 6371.0
        dlat = math.radians(r["lat"] - lat0)
        dlon = math.radians(r["lon"] - lon0)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat0))*math.cos(math.radians(r["lat"]))*math.sin(dlon/2)**2
        return 2 * R * math.asin(math.sqrt(a))
    df_top["km"] = df_top.apply(hav2, axis=1)
    df_top = df_top[df_top["km"] <= radius_km]

# Force numeric types
df_top["capacity"] = pd.to_numeric(df_top["capacity"], errors="coerce").fillna(0).astype(int)
df_top["risk"] = pd.to_numeric(df_top["risk"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

# Simple OR heuristic for suggested rebalancing based on expected deficit:
# expected_deficit ≈ risk * capacity * beta
BETA = 0.3
df_top["suggest_move"] = (df_top["risk"] * df_top["capacity"] * BETA).round().astype(int)

df_top["risk_pct"] = (df_top["risk"] * 100).round(2)
st.dataframe(
    df_top.sort_values("risk", ascending=False).head(top_n)[["station_id","name","capacity","risk_pct","suggest_move"]],
    use_container_width=True
)
st.markdown("---")

# ------------------------------
# 5) Model health (CloudWatch custom metrics)
# ------------------------------
st.subheader("3) Model Health — PR-AUC / F1 / Threshold hit-rate / Samples (T+30)")

st.info("🔍 **Real CloudWatch Mode** - Querying live metrics. If no data appears, ensure your MLOps pipeline is publishing metrics to the `Bikeshare/Model` namespace and your SageMaker endpoint is active.")

# Check endpoint status
with st.expander("🔍 System Status Check"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("SageMaker Endpoint")
        endpoint_active = check_endpoint_status()
    
    with col2:
        st.subheader("Custom Metrics Status")
        custom_metrics = list_available_metrics(CW_NS)
        if not custom_metrics.empty:
            st.success(f"✅ Found {len(custom_metrics)} custom metrics")
            for metric in custom_metrics['MetricName'].unique()[:5]:
                st.write(f"• {metric}")
        else:
            st.warning("⚠️ No custom metrics found in CloudWatch")
            st.write("Need to publish metrics from MLOps pipeline")

def cw_series(namespace: str, metric: str, dims: dict, minutes: int = 24*60, stat: str = "Average", demo_mode: bool = False) -> pd.DataFrame:
    """Fetch a time series from CloudWatch GetMetricData. Returns empty DataFrame if no data."""
    
    # Use mock data if demo mode is enabled
    if demo_mode:
        return generate_mock_metric_data(metric, hours=minutes//60)
    
    try:
        cw = cw_client()
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=minutes)
        
        # For SageMaker metrics, we need to be more flexible with dimensions
        # Try with just EndpointName first, then fall back to more specific dimensions
        dimension_list = [{"Name": k, "Value": v} for k, v in dims.items()]
        
        qry = [{
            "Id": "m1",
            "MetricStat": {
                "Metric": {
                    "Namespace": namespace,
                    "MetricName": metric,
                    "Dimensions": dimension_list,
                },
                "Period": 300,   # 5-minute resolution matches ops board
                "Stat": stat,
            },
            "ReturnData": True,
        }]
        
        resp = cw.get_metric_data(StartTime=start, EndTime=end, MetricDataQueries=qry, ScanBy="TimestampAscending")
        res = resp["MetricDataResults"][0]
        
        if not res.get("Timestamps"):
            # If no data with current dimensions, try with broader dimensions for SageMaker
            if namespace == "AWS/SageMaker" and len(dims) == 1 and "EndpointName" in dims:
                # Try again with VariantName=AllTraffic
                broader_dims = dims.copy()
                broader_dims["VariantName"] = "AllTraffic"
                return cw_series(namespace, metric, broader_dims, minutes, stat, demo_mode)
            return pd.DataFrame(columns=["ts", metric])
        ts, vs = res["Timestamps"], res["Values"]
        return pd.DataFrame({"ts": ts, metric: vs}).sort_values("ts")
    except Exception as e:
        if not demo_mode:
            st.error(f"CloudWatch error for {metric}: {str(e)}")
        return pd.DataFrame(columns=["ts", metric])

def plot_if_any(df: pd.DataFrame, x: str, y: str, title: str):
    """Plot a line if the DataFrame has data; hide completely if no data."""
    if not df.empty:
        fig = px.line(df, x=x, y=y, title=title)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    # If no data, don't show anything (clean interface)

dims_model = {"EndpointName": SM_ENDPOINT, "City": CITY}
dims_model_endpoint = {"EndpointName": SM_ENDPOINT}
dims_model_city = {"City": CITY}

# Try multiple dimension combinations for custom metrics
def get_custom_metric_flexible(metric_name: str) -> pd.DataFrame:
    """Try different dimension combinations to find custom metrics."""
    for dims in [dims_model, dims_model_endpoint, dims_model_city, {}]:
        result = cw_series(CW_NS, metric_name, dims, minutes=24*60, demo_mode=demo_mode)
        if not result.empty:
            return result
    return pd.DataFrame()

prauc = get_custom_metric_flexible("PR-AUC-24h")
f1 = get_custom_metric_flexible("F1-24h") 
hit = get_custom_metric_flexible("ThresholdHitRate-24h")
samp = get_custom_metric_flexible("Samples-24h")

plot_if_any(prauc, "ts", "PR-AUC-24h", "PR-AUC (24h rolling) — T+30")
plot_if_any(f1, "ts", "F1-24h", "F1 (24h rolling) — T+30")
plot_if_any(hit, "ts", "ThresholdHitRate-24h", "Threshold Hit Rate (24h) — T+30")
plot_if_any(samp, "ts", "Samples-24h", "Samples (24h) — T+30")

# --- NEW: PSI (feature drift) time series and a simple status badge ---
# Try to fetch the PSI metric (custom namespace Bikeshare/Model) with flexible dimensions
psi = get_custom_metric_flexible("PSI")

# Plot the 24h PSI line if available
plot_if_any(psi, "ts", "PSI", "PSI (24h) — Feature Drift")

# Show a quick status badge based on the latest PSI value (tune thresholds to your alarms)
if not psi.empty:
    latest_val = float(psi["PSI"].iloc[-1])
    # These bands align with common drift alarm practice: warn at 0.2, critical at 0.3
    if latest_val >= 0.30:
        st.error(f"Feature drift HIGH (PSI = {latest_val:.2f} ≥ 0.30)")
    elif latest_val >= 0.20:
        st.warning(f"Feature drift WARNING (PSI = {latest_val:.2f} ≥ 0.20)")
    else:
        st.success(f"Feature drift OK (PSI = {latest_val:.2f})")



st.markdown("---")

# ------------------------------
# 6) System health (SageMaker + custom heartbeat)
# ------------------------------
st.subheader("4) System Health — Endpoint latency / error rate / batch success")

# Use only EndpointName dimension for broad coverage; add VariantName if you split traffic.
dims_sm_endpoint = {"EndpointName": SM_ENDPOINT}

# Use 24-hour time windows for SageMaker metrics
lat_model = cw_series("AWS/SageMaker", "ModelLatency", dims_sm_endpoint, minutes=24*60, demo_mode=demo_mode)         # 24 hours
lat_over  = cw_series("AWS/SageMaker", "OverheadLatency", dims_sm_endpoint, minutes=24*60, demo_mode=demo_mode)
err4      = cw_series("AWS/SageMaker", "Invocation4XXErrors", dims_sm_endpoint, minutes=24*60, demo_mode=demo_mode)
err5      = cw_series("AWS/SageMaker", "Invocation5XXErrors", dims_sm_endpoint, minutes=24*60, demo_mode=demo_mode)

plot_if_any(lat_model, "ts", "ModelLatency", "ModelLatency (avg) — proxy for p50")
plot_if_any(lat_over, "ts", "OverheadLatency", "OverheadLatency (avg)")
plot_if_any(err4, "ts", "Invocation4XXErrors", "4XX Errors")
plot_if_any(err5, "ts", "Invocation5XXErrors", "5XX Errors")

# Batch success proxy via custom metric (PredictionHeartbeat == 1)
# Try different dimension combinations for custom metrics
hb_dims_full = {"EndpointName": SM_ENDPOINT, "City": CITY}
hb_dims_endpoint = {"EndpointName": SM_ENDPOINT}
hb_dims_city = {"City": CITY}

hb = cw_series(CW_NS, "PredictionHeartbeat", hb_dims_full, minutes=24*60, stat="Sum", demo_mode=demo_mode)
if hb.empty:
    hb = cw_series(CW_NS, "PredictionHeartbeat", hb_dims_endpoint, minutes=24*60, stat="Sum", demo_mode=demo_mode)
if hb.empty:
    hb = cw_series(CW_NS, "PredictionHeartbeat", hb_dims_city, minutes=24*60, stat="Sum", demo_mode=demo_mode)
if hb.empty:
    hb = cw_series(CW_NS, "PredictionHeartbeat", {}, minutes=24*60, stat="Sum", demo_mode=demo_mode)

plot_if_any(hb, "ts", "PredictionHeartbeat", "Prediction heartbeat (Sum)")


st.markdown("---")

# ------------------------------
# 7) Data Freshness — partitions & delays
# ------------------------------
st.subheader("5) Data Freshness — latest partitions and delay (min)")

# NOTE: your dt format is 'YYYY-MM-DD-HH-mm' (string). Use max(dt) and then parse separately.
fresh_sql_tpl = """
SELECT '{table}' AS source,
       max(dt) AS latest_dt_str
FROM {db}.{table}
WHERE city = '{city}'
"""

sources = [TBL_STATUS, TBL_INFO, TBL_WEATHER, TBL_FEATURES, TBL_INFER, TBL_MON_QUAL]
records = []
now_utc = datetime.now(timezone.utc)

for tbl in sources:
    try:
        df_latest = run_athena(fresh_sql_tpl.format(table=tbl, db=DB, city=CITY))
        if not df_latest.empty and pd.notnull(df_latest.loc[0, "latest_dt_str"]):
            latest_str = df_latest.loc[0, "latest_dt_str"]
            # Parse the dt string manually: "2025-10-03-02-00" -> datetime
            try:
                latest = datetime.strptime(latest_str, "%Y-%m-%d-%H-%M").replace(tzinfo=timezone.utc)
                delay_min = (now_utc - latest).total_seconds() / 60.0
                records.append({"source": tbl, "latest_utc": latest, "delay_min": round(delay_min, 1)})
            except ValueError:
                records.append({"source": tbl, "latest_utc": latest_str, "delay_min": "parse_error"})
        else:
            records.append({"source": tbl, "latest_utc": None, "delay_min": None})
    except Exception as e:
        # If a table does not have 'city' or 'dt' (schema variance), keep it safe and visible.
        records.append({"source": tbl, "latest_utc": f"error: {str(e)}", "delay_min": None})

df_fresh = pd.DataFrame(records)
st.dataframe(df_fresh, use_container_width=True)
st.caption("Targets: ingestion every 5 min; data lake latency ≤ 3 min; quality written hourly.")

# ------------------------------
# 7.5) CloudWatch Metrics Setup Guide
# ------------------------------
with st.expander("💡 How to Publish CloudWatch Metrics"):
    st.markdown("""
    **To see real data in this dashboard, publish these metrics from your MLOps pipeline:**
    
    ```python
    import boto3
    
    cloudwatch = boto3.client('cloudwatch')
    
    # Example: Publish model performance metrics
    cloudwatch.put_metric_data(
        Namespace='Bikeshare/Model',
        MetricData=[
            {
                'MetricName': 'PR-AUC-24h',
                'Value': 0.85,  # Your calculated PR-AUC value
                'Dimensions': [
                    {'Name': 'EndpointName', 'Value': 'bikeshare-prod'},
                    {'Name': 'City', 'Value': 'nyc'}
                ]
            },
            {
                'MetricName': 'F1-24h', 
                'Value': 0.78,  # Your calculated F1 score
                'Dimensions': [
                    {'Name': 'EndpointName', 'Value': 'bikeshare-prod'},
                    {'Name': 'City', 'Value': 'nyc'}
                ]
            },
            {
                'MetricName': 'PredictionHeartbeat',
                'Value': 1,  # 1 for success, 0 for failure
                'Dimensions': [
                    {'Name': 'EndpointName', 'Value': 'bikeshare-prod'},
                    {'Name': 'City', 'Value': 'nyc'}
                ]
            }
        ]
    )
    ```
    
    **SageMaker metrics are automatically published when your endpoint receives traffic.**
    """)
    
    # Quick test button
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧪 Test CloudWatch Connection"):
            try:
                cloudwatch = cw_client()
                # Try to put a test metric
                cloudwatch.put_metric_data(
                    Namespace=CW_NS,
                    MetricData=[
                        {
                            'MetricName': 'TestConnection',
                            'Value': 1.0,
                            'Dimensions': [
                                {'Name': 'Test', 'Value': 'Connection'}
                            ]
                        }
                    ]
                )
                st.success("✅ Successfully connected to CloudWatch!")
            except Exception as e:
                st.error(f"❌ Failed to connect: {str(e)}")
    
    with col2:
        if st.button("📊 Publish Sample Metrics"):
            try:
                import random
                cloudwatch = cw_client()
                
                # 发布一组样本指标
                sample_metrics = [
                    {
                        'MetricName': 'PR-AUC-24h',
                        'Value': round(0.80 + random.uniform(0, 0.15), 3),
                        'Dimensions': [
                            {'Name': 'EndpointName', 'Value': SM_ENDPOINT},
                            {'Name': 'City', 'Value': CITY}
                        ]
                    },
                    {
                        'MetricName': 'F1-24h',
                        'Value': round(0.70 + random.uniform(0, 0.20), 3),
                        'Dimensions': [
                            {'Name': 'EndpointName', 'Value': SM_ENDPOINT},
                            {'Name': 'City', 'Value': CITY}
                        ]
                    },
                    {
                        'MetricName': 'PredictionHeartbeat',
                        'Value': 1,
                        'Dimensions': [
                            {'Name': 'EndpointName', 'Value': SM_ENDPOINT},
                            {'Name': 'City', 'Value': CITY}
                        ]
                    }
                ]
                
                cloudwatch.put_metric_data(
                    Namespace=CW_NS,
                    MetricData=sample_metrics
                )
                
                st.success("✅ Sample metrics published!")
                st.info("🔄 Refresh the page in 2-3 minutes to see the data")
                
            except Exception as e:
                st.error(f"❌ Failed to publish metrics: {str(e)}")

# ------------------------------
# 8) Footer: debug timings (helps verify <3s)
# ------------------------------
with st.expander("Debug: Athena query timings (ms)"):
    st.write(st.session_state.get("_debug_last_query_ms", {}))

# ------------------------------
# 9) CloudWatch metrics debugging
# ------------------------------
with st.expander("Debug: Available CloudWatch Metrics"):
    st.markdown("""
    **Expected Custom Metrics for Real Mode:**
    - `PR-AUC-24h` - Model performance metric
    - `F1-24h` - Model performance metric  
    - `ThresholdHitRate-24h` - Model performance metric
    - `Samples-24h` - Data volume metric
    - `PredictionHeartbeat` - System health metric
    
    **Expected SageMaker Metrics:**
    - `ModelLatency` - Inference latency
    - `OverheadLatency` - Infrastructure latency
    - `Invocation4XXErrors` - Client errors
    - `Invocation5XXErrors` - Server errors
    """)
    
    st.subheader("Custom Metrics (Bikeshare/Model)")
    custom_metrics = list_available_metrics(CW_NS)
    if not custom_metrics.empty:
        st.dataframe(custom_metrics, use_container_width=True)
    else:
        st.info("No custom metrics found - check if metrics are being published")
    
    st.subheader("SageMaker Metrics")
    sm_metrics = list_available_metrics("AWS/SageMaker")
    if not sm_metrics.empty:
        # Filter to show only metrics for our endpoint
        endpoint_metrics = sm_metrics[sm_metrics['Dimensions'].str.contains(SM_ENDPOINT, na=False)]
        if not endpoint_metrics.empty:
            st.dataframe(endpoint_metrics, use_container_width=True)
        else:
            st.info(f"No SageMaker metrics found for endpoint: {SM_ENDPOINT}")
            st.write("All SageMaker metrics:")
            st.dataframe(sm_metrics.head(10), use_container_width=True)
    else:
        st.info("No SageMaker metrics found")
