"""Dashboard rendering functions."""
from __future__ import annotations

from datetime import datetime, timezone
from math import ceil

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from .contracts import ArtifactLoadResult, FreshnessLoadResult, LoadStatus
from .presentation import (
    MetricSpec,
    build_data_status_frame,
    classify_metric_status,
    critical_threshold_for,
    format_utc_label,
    metric_empty_message,
    station_history_title,
    summarize_quality_availability,
)
from .targeting import DashboardTargetConfig

_CLR_CRITICAL = "#c1121f"
_CLR_ALERT = "#f77f00"
_CLR_OK = "#2a9d8f"
_CLR_INFO = "#577590"
_CLR_LINE = "#264653"

_METRIC_LABELS: dict[str, str] = {
    "PR-AUC-24h": "PR-AUC (24h)",
    "F1-24h": "F1 (24h)",
    "PredictionHeartbeat": "Prediction Heartbeat (24h)",
    "ThresholdHitRate-24h": "Threshold Hit Rate (24h)",
    "Samples-24h": "Samples (24h)",
    "ModelLatency": "ModelLatency p95 (ms)",
    "Invocation5XXErrors": "Invocation5XXErrors (24h)",
    "Invocation4XXErrors": "Invocation4XXErrors (24h)",
    "Invocations": "Invocations (24h)",
    "PSI": "PSI (24h)",
}

_ENV_BADGES: dict[str, str] = {
    "production": "PRODUCTION",
    "staging": "STAGING",
    "local": "LOCAL",
}

_STATUS_MESSAGES = {
    LoadStatus.NO_OBJECTS: lambda source: f"{source} has not been published yet.",
    LoadStatus.ALL_SCORES_NULL: lambda source: f"{source} exists, but all prediction scores are invalid. Upstream inference likely failed.",
    LoadStatus.ACCESS_DENIED: lambda source: f"{source} cannot be read because AWS access is denied.",
    LoadStatus.READ_ERROR: lambda source: f"{source} could not be read. Check permissions, network, or artifact integrity.",
    LoadStatus.SCHEMA_ERROR: lambda source: f"{source} does not match the dashboard contract.",
}


def _render_status_chip(text: str, color: str) -> None:
    st.markdown(
        (
            f"<div style='display:inline-block;padding:2px 10px;border-radius:999px;"
            f"background:{color}18;color:{color};font-weight:600;font-size:0.78rem;"
            f"margin:0 0 0.45rem 0'>{text}</div>"
        ),
        unsafe_allow_html=True,
    )


def _format_metric_value(value: float, decimals: int) -> str:
    if abs(value) >= 1000:
        return f"{value:.0f}"
    return f"{value:.{decimals}f}"


def render_status_cards(
    *,
    target: DashboardTargetConfig,
    environment: str,
    pipeline_state: str,
    endpoint_name: str,
    model_version: str,
    latest_prediction_dt: datetime | None,
    threshold: float,
) -> None:
    badge_text = _ENV_BADGES.get(environment.lower(), environment.upper())
    badge_color = {
        "production": _CLR_OK,
        "staging": _CLR_ALERT,
        "local": "#888888",
    }.get(environment.lower(), _CLR_INFO)

    st.markdown(
        (
            f"<div style='display:inline-block;padding:4px 14px;border-radius:20px;"
            f"background:{badge_color};color:white;font-weight:600;font-size:0.85rem;"
            f"margin-bottom:0.6rem'>{badge_text}</div>"
        ),
        unsafe_allow_html=True,
    )

    latest_label = latest_prediction_dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if latest_prediction_dt else "Unavailable"
    cols = st.columns(5)
    cols[0].metric("Pipeline state", pipeline_state)
    cols[1].metric("Prediction target", target.display_name)
    cols[2].metric("Active endpoint", endpoint_name)
    cols[3].metric("Active model", model_version)
    cols[4].metric("Latest prediction", latest_label)
    st.caption(f"City: paris | Environment: {environment} | Alert threshold: {threshold:.2f}")


def render_artifact_issue(result: ArtifactLoadResult, *, default_message: str | None = None) -> None:
    if result.status == LoadStatus.OK:
        return
    message = result.message or _STATUS_MESSAGES.get(result.status, lambda source: default_message or "Data unavailable.")(result.source_name or "Artifact")
    if result.status in {LoadStatus.ALL_SCORES_NULL, LoadStatus.ACCESS_DENIED, LoadStatus.READ_ERROR, LoadStatus.SCHEMA_ERROR}:
        st.error(message)
    elif result.status == LoadStatus.NO_OBJECTS:
        st.warning(message)
    else:
        st.info(default_message or message)


def render_alert_banner(
    *,
    prediction_result: ArtifactLoadResult,
    target: DashboardTargetConfig,
    threshold: float,
    stale_after_minutes: int,
) -> None:
    if prediction_result.status != LoadStatus.OK:
        render_artifact_issue(prediction_result)
        return

    predictions = prediction_result.data
    latest_dt = prediction_result.latest_dt
    if latest_dt is not None:
        age_minutes = (datetime.now(timezone.utc) - latest_dt).total_seconds() / 60
        if age_minutes > stale_after_minutes:
            st.warning(
                f"Latest {target.display_name.lower()} predictions are stale by {age_minutes:.1f} minutes. "
                "Investigate the upstream prediction pipeline before trusting the map."
            )
            return

    critical_threshold = critical_threshold_for(threshold)
    n_critical = int((predictions["score"] >= critical_threshold).sum())
    n_alert = int(((predictions["score"] >= threshold) & (predictions["score"] < critical_threshold)).sum())
    n_total = len(predictions)

    if n_critical > 0:
        st.error(
            f"{n_critical} stations are at critical {target.display_name.lower()} risk. "
            f"{n_alert} additional stations are above the alert threshold across {n_total} scored stations."
        )
    elif n_alert > 0:
        st.warning(
            f"{n_alert} stations are above the alert threshold for {target.display_name.lower()}. "
            f"{n_total - n_alert} stations remain below threshold."
        )
    else:
        st.success(f"All {n_total} scored stations are below the configured alert threshold.")


def render_prediction_map(
    *,
    station_risk_frame: pd.DataFrame,
    target: DashboardTargetConfig,
    threshold: float,
) -> None:
    st.subheader(f"Live operations map | {target.display_name} | next 30 minutes (UTC)")
    if station_risk_frame.empty:
        st.warning("Business-ready station risk data is unavailable.")
        return

    center = [station_risk_frame["lat"].mean(), station_risk_frame["lon"].mean()]
    fmap = folium.Map(location=center, zoom_start=12, tiles="CartoDB Positron")

    for row in station_risk_frame.itertuples():
        tooltip = f"{row.station_id} | {row.station_name}"
        popup_html = (
            f"<b>{row.station_name}</b><br>"
            f"Station ID: {row.station_id}<br>"
            f"Current bikes: {int(row.bikes) if pd.notna(row.bikes) else 'n/a'}<br>"
            f"Current docks: {int(row.docks) if pd.notna(row.docks) else 'n/a'}<br>"
            f"Capacity: {int(row.capacity) if pd.notna(row.capacity) else 'n/a'}<br>"
            f"Current status: {row.current_status}<br>"
            f"{target.display_name} risk level: {row.risk_level}<br>"
            f"{target.display_name} probability (30 min): {row.stockout_probability:.3f}<br>"
            f"Last updated: {format_utc_label(row.ts)}"
        )
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=6,
            color=row.risk_color,
            fill=True,
            fill_color=row.risk_color,
            fill_opacity=0.85,
            tooltip=tooltip,
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(fmap)

    event = st_folium(fmap, width=None, height=440, returned_objects=["last_object_clicked_tooltip"])
    st.markdown(
        (
            f"<span style='color:{_CLR_CRITICAL}'>●</span> Critical "
            f"<span style='color:{_CLR_ALERT};margin-left:16px'>●</span> Alert "
            f"<span style='color:{_CLR_OK};margin-left:16px'>●</span> Normal"
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        f"Selected target uses a model-specific alert threshold of {threshold:.2f}. "
        "Click a station to keep that station selected across tabs."
    )

    clicked = event.get("last_object_clicked_tooltip") if isinstance(event, dict) else None
    if clicked and " | " in clicked:
        station_id, station_name = clicked.split(" | ", 1)
        st.session_state["selected_station_id"] = station_id.strip()
        st.session_state["selected_station_name"] = station_name.strip()


def render_selected_station_summary(
    *,
    selected_station: dict[str, object] | None,
    target: DashboardTargetConfig,
) -> None:
    st.subheader("Selected station summary | next 30 minutes (UTC)")
    if selected_station is None:
        st.info("No station is available yet. Once predictions load, the highest-risk station will be selected automatically.")
        return

    station_name = str(selected_station.get("station_name") or selected_station.get("station_id"))
    station_id = str(selected_station.get("station_id"))
    st.markdown(f"**{station_name}**  |  Station ID: `{station_id}`")

    cols = st.columns(6)
    cols[0].metric("Current bikes", int(selected_station["bikes"]) if pd.notna(selected_station.get("bikes")) else "n/a")
    cols[1].metric("Current docks", int(selected_station["docks"]) if pd.notna(selected_station.get("docks")) else "n/a")
    cols[2].metric("Capacity", int(selected_station["capacity"]) if pd.notna(selected_station.get("capacity")) else "n/a")
    cols[3].metric("Current status", str(selected_station.get("current_status") or "Unavailable"))
    cols[4].metric(f"{target.display_name} risk", str(selected_station.get("risk_level") or "Unavailable"))
    probability = float(selected_station["stockout_probability"]) if pd.notna(selected_station.get("stockout_probability")) else None
    cols[5].metric("30-minute probability", f"{probability:.3f}" if probability is not None else "n/a")
    st.caption(f"Latest serving snapshot: {format_utc_label(selected_station.get('dt') or selected_station.get('ts'))}")


def render_top_risk_table(
    *,
    station_risk_frame: pd.DataFrame,
    top_n: int,
) -> None:
    st.subheader(f"Station risk table | top {top_n} stations | next 30 minutes (UTC)")
    if station_risk_frame.empty:
        st.warning("Risk table is unavailable.")
        return

    ranked = station_risk_frame.head(top_n).copy()
    ranked = ranked.rename(
        columns={
            "station_name": "Station name",
            "station_id": "Station ID",
            "bikes": "Current bikes",
            "docks": "Current docks",
            "capacity": "Capacity",
            "current_status": "Current status",
            "stockout_probability": "30-minute stockout probability",
            "ts": "Last updated",
        }
    )

    st.dataframe(
        ranked[
            [
                "Station name",
                "Station ID",
                "Current bikes",
                "Current docks",
                "Capacity",
                "Current status",
                "30-minute stockout probability",
                "Last updated",
            ]
        ],
        width="stretch",
        column_config={
            "30-minute stockout probability": st.column_config.ProgressColumn(
                label="30-minute stockout probability",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            ),
            "Current bikes": st.column_config.NumberColumn(format="%d"),
            "Current docks": st.column_config.NumberColumn(format="%d"),
            "Capacity": st.column_config.NumberColumn(format="%d"),
        },
        hide_index=True,
    )


def render_history_chart(
    *,
    history_result: ArtifactLoadResult,
    target: DashboardTargetConfig,
    threshold: float,
    selected_station: dict[str, object] | None,
) -> None:
    st.subheader(station_history_title(selected_station))
    st.caption(
        "30-minute stockout definition: min bikes/docks in the next 30 minutes <= 2 | "
        "Alert threshold: model-specific operating threshold learned during training."
    )
    if history_result.status != LoadStatus.OK:
        render_artifact_issue(history_result, default_message="History is unavailable.")
        return

    df = history_result.data.sort_values("ts")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["ts"],
            y=df["score"],
            mode="lines+markers",
            name=target.display_name,
            line=dict(color=_CLR_LINE, width=2),
            marker=dict(size=5),
        )
    )
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=_CLR_ALERT,
        annotation_text=f"Target-specific alert threshold ({threshold:.2f})",
        annotation_position="bottom right",
    )
    fig.update_layout(
        xaxis_title="Time (UTC)",
        yaxis_title="30-minute stockout probability",
        yaxis=dict(range=[0, 1.05]),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")


def render_quality_status_panel(
    *,
    quality_result: ArtifactLoadResult,
    metric_series_map: dict[str, pd.DataFrame],
) -> None:
    severity, title, message = summarize_quality_availability(
        quality_result=quality_result,
        metric_series_map=metric_series_map,
    )
    body = f"{message} Latest quality artifact timestamp: {format_utc_label(quality_result.latest_dt)}."
    if severity == "success":
        st.success(f"{title} {body}")
    elif severity == "warning":
        st.warning(f"{title} {body}")
    elif severity == "error":
        st.error(f"{title} {body}")
    else:
        st.info(f"{title} {body}")
    st.caption("Expected lag: 30-minute label maturity + 7-minute backfill start delay, so quality evidence normally trails predictions.")


def render_metric_section(
    *,
    title: str,
    series_map: dict[str, pd.DataFrame],
    metric_specs: dict[str, MetricSpec] | None = None,
    human_labels: dict[str, str] | None = None,
    quality_result: ArtifactLoadResult | None = None,
    cols_per_row: int = 3,
) -> None:
    st.subheader(title)
    labels = {**_METRIC_LABELS, **(human_labels or {})}
    specs = metric_specs or {}
    items = list(series_map.items())
    if not items:
        st.info("No metrics configured for this section.")
        return

    for row_idx in range(ceil(len(items) / cols_per_row)):
        row_items = items[row_idx * cols_per_row : (row_idx + 1) * cols_per_row]
        cols = st.columns(len(row_items))

        for col, (metric_name, series) in zip(cols, row_items):
            spec = specs.get(metric_name)
            display_name = labels.get(metric_name, metric_name)
            with col:
                if series.empty or metric_name not in series.columns:
                    _render_status_chip("Unavailable", _CLR_INFO)
                    st.warning(
                        f"{display_name}: "
                        f"{metric_empty_message(metric_name=metric_name, spec=spec, quality_result=quality_result)}"
                    )
                    continue

                current = float(series[metric_name].iloc[-1])
                prev = float(series[metric_name].iloc[-2]) if len(series) > 1 else current
                delta = current - prev
                status_text, status_color = classify_metric_status(current, spec or MetricSpec(label=display_name))
                _render_status_chip(status_text, status_color)
                decimals = spec.decimals if spec is not None else 3

                st.metric(
                    label=display_name,
                    value=_format_metric_value(current, decimals),
                    delta=_format_metric_value(delta, decimals) if delta != 0 else f"{delta:+.{decimals}f}",
                )

                fig = go.Figure(
                    go.Scatter(
                        x=series["ts"],
                        y=series[metric_name],
                        mode="lines",
                        line=dict(color=_CLR_LINE, width=2),
                        fill="tozeroy",
                        fillcolor="rgba(38,70,83,0.08)",
                    )
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
                    showlegend=False,
                    height=160,
                )
                st.plotly_chart(fig, width="stretch")


def render_data_status_table(
    *,
    prediction_result: ArtifactLoadResult,
    quality_result: ArtifactLoadResult,
    freshness_result: FreshnessLoadResult,
    prediction_sla_minutes: int,
    quality_sla_minutes: int,
    feature_sla_minutes: int,
) -> None:
    st.subheader("Data pipeline status | operator SLA view")
    frame = build_data_status_frame(
        prediction_result=prediction_result,
        quality_result=quality_result,
        freshness_result=freshness_result,
        prediction_sla_minutes=prediction_sla_minutes,
        quality_sla_minutes=quality_sla_minutes,
        feature_sla_minutes=feature_sla_minutes,
    )
    if frame.empty:
        st.warning("No data status rows are available.")
        return
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config={"Delay (min)": st.column_config.NumberColumn(format="%.1f min")},
    )
