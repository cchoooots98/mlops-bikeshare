"""Dashboard rendering functions."""

from __future__ import annotations

import re
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
    PREDICTION_ARTIFACT_POLICY,
    assess_schedule_freshness,
    build_data_status_frame,
    classify_metric_status,
    critical_threshold_for,
    format_utc_label,
    metric_empty_message,
    station_history_context,
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
    "PSI": "PSI Overall (24h)",
    "PSI_core": "PSI Core (24h)",
    "PSI_weather": "PSI Weather (24h)",
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


def _coerce_int(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "n/a"
    return str(int(numeric))


def _slugify_key_part(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "section"


def _resolve_metric_summary(
    series: pd.DataFrame, metric_name: str, spec: MetricSpec | None
) -> tuple[float | None, float | None]:
    if metric_name not in series.columns:
        return None, None

    values = pd.to_numeric(series[metric_name], errors="coerce").dropna()
    if values.empty:
        return None, None

    if spec is not None and spec.summary == "window_sum":
        return float(values.sum()), None

    current = float(values.iloc[-1])
    previous = float(values.iloc[-2]) if len(values) > 1 else current
    return current, current - previous


def _format_detail_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "-"

    normalized = text.replace("\\", "/")
    if "/" in normalized:
        parts = [part for part in normalized.split("/") if part]
        if len(parts) >= 5:
            return "/".join([*parts[:3], "...", *parts[-2:]])
        if len(parts) == 4:
            return "/".join([*parts[:2], "...", parts[-1]])

    if len(normalized) <= 96:
        return normalized
    return f"{normalized[:52]}...{normalized[-28:]}"


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

    latest_label = (
        latest_prediction_dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        if latest_prediction_dt
        else "Unavailable"
    )
    cols = st.columns(2)
    cols[0].metric("Pipeline state", pipeline_state)
    cols[1].metric("Prediction target", target.display_name)
    st.markdown(
        (
            "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));"
            "gap:0.75rem;margin-top:0.6rem;margin-bottom:0.25rem'>"
            f"<div style='padding:0.7rem 0.9rem;border:1px solid #e9ecef;border-radius:8px;background:#fbfcfd'>"
            f"<div style='font-size:0.74rem;color:#6c757d;margin-bottom:0.2rem'>Active endpoint</div>"
            f"<div style='font-size:0.95rem;font-weight:600;line-height:1.35;word-break:break-word'>{endpoint_name}</div>"
            "</div>"
            f"<div style='padding:0.7rem 0.9rem;border:1px solid #e9ecef;border-radius:8px;background:#fbfcfd'>"
            f"<div style='font-size:0.74rem;color:#6c757d;margin-bottom:0.2rem'>Active model</div>"
            f"<div style='font-size:0.95rem;font-weight:600;line-height:1.35;word-break:break-word'>{model_version}</div>"
            "</div>"
            f"<div style='padding:0.7rem 0.9rem;border:1px solid #e9ecef;border-radius:8px;background:#fbfcfd'>"
            f"<div style='font-size:0.74rem;color:#6c757d;margin-bottom:0.2rem'>Latest prediction</div>"
            f"<div style='font-size:0.95rem;font-weight:600;line-height:1.35;word-break:break-word'>{latest_label}</div>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_artifact_issue(
    result: ArtifactLoadResult, *, default_message: str | None = None
) -> None:
    if result.status == LoadStatus.OK:
        return
    message = result.message or _STATUS_MESSAGES.get(
        result.status, lambda source: default_message or "Data unavailable."
    )(result.source_name or "Artifact")
    if result.status in {
        LoadStatus.ALL_SCORES_NULL,
        LoadStatus.ACCESS_DENIED,
        LoadStatus.READ_ERROR,
        LoadStatus.SCHEMA_ERROR,
    }:
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
        assessment = assess_schedule_freshness(
            latest_dt=latest_dt,
            policy=PREDICTION_ARTIFACT_POLICY,
            now_utc=datetime.now(timezone.utc),
        )
        if assessment.status == "Critical":
            st.error(
                f"Latest {target.display_name.lower()} predictions are {assessment.excess_delay_minutes:.1f} minutes "
                "behind the current serving schedule. Investigate the upstream prediction pipeline before trusting the map."
            )
            return
        if assessment.status == "Warning":
            st.warning(
                f"Latest {target.display_name.lower()} predictions are {assessment.excess_delay_minutes:.1f} minutes "
                "behind the current serving schedule. Investigate the upstream prediction pipeline before trusting the map."
            )
            return

    critical_threshold = critical_threshold_for(threshold)
    n_critical = int((predictions["score"] >= critical_threshold).sum())
    n_alert = int(
        (
            (predictions["score"] >= threshold)
            & (predictions["score"] < critical_threshold)
        ).sum()
    )
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
        st.success(
            f"All {n_total} scored stations are below the configured alert threshold."
        )


def render_prediction_map(
    *,
    station_risk_frame: pd.DataFrame,
    target: DashboardTargetConfig,
    threshold: float,
) -> None:
    st.subheader("Live Operations Map")
    st.caption(f"{target.display_name} risk for the next 30 minutes (UTC).")
    if station_risk_frame.empty:
        st.warning("Business-ready station risk data is unavailable.")
        return

    center = [station_risk_frame["lat"].mean(), station_risk_frame["lon"].mean()]
    fmap = folium.Map(location=center, zoom_start=12, tiles="CartoDB Positron")

    for row in station_risk_frame.itertuples():
        tooltip = f"{row.station_id} | {row.station_name}"
        current_inventory_label = (
            "Current bikes" if target.target_name == "bikes" else "Current docks"
        )
        current_inventory_value = (
            row.bikes if target.target_name == "bikes" else row.docks
        )
        popup_html = (
            "<div style='font-size:0.98rem;line-height:1.3'>"
            f"<div style='font-weight:700;margin-bottom:0.2rem'>{row.station_name}</div>"
            f"<div><b>ID:</b> {row.station_id}</div>"
            f"<div><b>{current_inventory_label}:</b> {_coerce_int(current_inventory_value)}</div>"
            f"<div><b>Capacity:</b> {_coerce_int(row.capacity)}</div>"
            f"<div><b>Status:</b> {row.current_status}</div>"
            f"<div><b>Risk:</b> {row.risk_level} ({row.stockout_probability:.3f})</div>"
            f"<div><b>Updated:</b> {format_utc_label(row.ts)}</div>"
            "</div>"
        )
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=6,
            color=row.risk_color,
            fill=True,
            fill_color=row.risk_color,
            fill_opacity=0.85,
            tooltip=tooltip,
            popup=folium.Popup(popup_html, max_width=240),
        ).add_to(fmap)

    event = st_folium(
        fmap, width=None, height=440, returned_objects=["last_object_clicked_tooltip"]
    )
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

    clicked = (
        event.get("last_object_clicked_tooltip") if isinstance(event, dict) else None
    )
    if clicked and " | " in clicked:
        station_id, station_name = clicked.split(" | ", 1)
        st.session_state["selected_station_id"] = station_id.strip()
        st.session_state["selected_station_name"] = station_name.strip()


def render_selected_station_summary(
    *,
    selected_station: dict[str, object] | None,
    target: DashboardTargetConfig,
) -> None:
    st.subheader("Selected Station Summary")
    if selected_station is None:
        st.info(
            "No station is available yet. Once predictions load, the highest-risk station will be selected automatically."
        )
        return

    station_name = str(
        selected_station.get("station_name") or selected_station.get("station_id")
    )
    station_id = str(selected_station.get("station_id"))
    st.markdown(
        f"<div style='font-size:1.35rem;font-weight:700;line-height:1.3;margin-bottom:0.2rem'>{station_name}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Station ID: {station_id}. Forecast horizon: next 30 minutes (UTC).")
    probability = (
        float(selected_station["stockout_probability"])
        if pd.notna(selected_station.get("stockout_probability"))
        else None
    )
    summary_cards = [
        ("Current bikes", _coerce_int(selected_station.get("bikes"))),
        ("Current docks", _coerce_int(selected_station.get("docks"))),
        ("Capacity", _coerce_int(selected_station.get("capacity"))),
        (
            "Current status",
            str(selected_station.get("current_status") or "Unavailable"),
        ),
        (
            f"{target.display_name} risk",
            str(selected_station.get("risk_level") or "Unavailable"),
        ),
        (
            "30-minute probability",
            f"{probability:.3f}" if probability is not None else "n/a",
        ),
    ]
    summary_html = "".join(
        (
            "<div style='padding:0.9rem 1rem;border:1px solid #e9ecef;border-radius:10px;background:#fbfcfd;'>"
            f"<div style='font-size:0.78rem;color:#6c757d;margin-bottom:0.35rem'>{label}</div>"
            f"<div style='font-size:1.05rem;font-weight:600;line-height:1.35;word-break:break-word'>{value}</div>"
            "</div>"
        )
        for label, value in summary_cards
    )
    st.markdown(
        (
            "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));"
            f"gap:0.75rem;margin:0.5rem 0 0.25rem'>{summary_html}</div>"
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        f"Latest serving snapshot: {format_utc_label(selected_station.get('dt') or selected_station.get('ts'))}"
    )


def render_top_risk_table(
    *,
    station_risk_frame: pd.DataFrame,
    top_n: int,
    target: DashboardTargetConfig,
) -> None:
    st.subheader(f"Top {top_n} Stations by 30-Minute Risk")
    st.caption(
        f"{target.display_name} view with current inventory and the latest serving snapshot (UTC)."
    )
    if station_risk_frame.empty:
        st.warning("Risk table is unavailable.")
        return

    ranked = (
        station_risk_frame.loc[
            pd.to_numeric(station_risk_frame["capacity"], errors="coerce")
            .fillna(0)
            .gt(0)
        ]
        .head(top_n)
        .copy()
    )
    if ranked.empty:
        st.warning(
            "No stations with positive capacity are available for the ranked table."
        )
        return

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
                "Station ID",
                "Station name",
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
    st.subheader(station_history_title())
    st.caption(station_history_context(selected_station))
    st.caption(
        "30-minute stockout definition: min bikes/docks in the next 30 minutes <= 2. "
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
    station_id = str((selected_station or {}).get("station_id") or "none")
    st.plotly_chart(
        fig,
        width="stretch",
        key=f"history-chart-{_slugify_key_part(target.target_name)}-{_slugify_key_part(station_id)}",
    )


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
    st.caption(
        "Expected lag: 30-minute label maturity plus the next scheduled quality backfill cycle, so quality evidence normally trails predictions."
    )


def render_metric_section(
    *,
    title: str,
    series_map: dict[str, pd.DataFrame],
    metric_specs: dict[str, MetricSpec] | None = None,
    human_labels: dict[str, str] | None = None,
    quality_result: ArtifactLoadResult | None = None,
    cols_per_row: int = 3,
    description: str | None = None,
    key_prefix: str | None = None,
) -> None:
    st.subheader(title)
    if description:
        st.caption(description)
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
            display_name = (
                spec.label if spec is not None else labels.get(metric_name, metric_name)
            )
            with col:
                if series.empty or metric_name not in series.columns:
                    _render_status_chip("Unavailable", _CLR_INFO)
                    st.warning(
                        f"{display_name}: "
                        f"{metric_empty_message(metric_name=metric_name, spec=spec, quality_result=quality_result)}"
                    )
                    continue

                current, delta = _resolve_metric_summary(series, metric_name, spec)
                if current is None:
                    _render_status_chip("Unavailable", _CLR_INFO)
                    st.warning(
                        f"{display_name}: "
                        f"{metric_empty_message(metric_name=metric_name, spec=spec, quality_result=quality_result)}"
                    )
                    continue
                status_text, status_color = classify_metric_status(
                    current, spec or MetricSpec(label=display_name)
                )
                _render_status_chip(status_text, status_color)
                decimals = spec.decimals if spec is not None else 3

                metric_kwargs = {
                    "label": display_name,
                    "value": _format_metric_value(current, decimals),
                }
                if delta is not None:
                    metric_kwargs["delta"] = (
                        _format_metric_value(delta, decimals)
                        if delta != 0
                        else f"{delta:+.{decimals}f}"
                    )
                st.metric(**metric_kwargs)

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
                chart_key_prefix = key_prefix or _slugify_key_part(title)
                chart_key = f"metric-chart-{_slugify_key_part(chart_key_prefix)}-{_slugify_key_part(metric_name)}"
                st.plotly_chart(fig, width="stretch", key=chart_key)


def render_data_status_table(
    *,
    prediction_result: ArtifactLoadResult,
    quality_result: ArtifactLoadResult,
    freshness_result: FreshnessLoadResult,
    prediction_sla_minutes: int,
    quality_sla_minutes: int,
    feature_sla_minutes: int,
) -> None:
    st.subheader("Data Pipeline Status")
    st.caption("Source freshness plus schedule-aware pipeline freshness, with missed-cycle severity for delayed stages.")
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
    display_frame = frame.copy()
    display_frame["Last updated (UTC)"] = display_frame["Last updated (UTC)"].apply(
        format_utc_label
    )
    display_frame["Delay (min)"] = display_frame["Delay (min)"].apply(
        lambda value: f"{float(value):.1f} min" if pd.notna(value) else "n/a"
    )
    display_frame["Expected lag (min)"] = display_frame["Expected lag (min)"].apply(
        lambda value: f"{float(value):.1f} min" if pd.notna(value) else "n/a"
    )
    display_frame["Excess lag (min)"] = display_frame["Excess lag (min)"].apply(
        lambda value: f"{float(value):.1f} min" if pd.notna(value) else "n/a"
    )
    display_frame["Details"] = display_frame["Details"].apply(_format_detail_text)
    compact_frame = (
        display_frame[
            [
                "Data source",
                "Last updated (UTC)",
                "Delay (min)",
                "Expected lag (min)",
                "Excess lag (min)",
                "Status",
            ]
        ]
        .rename(
            columns={
                "Last updated (UTC)": "Updated (UTC)",
                "Delay (min)": "Lag",
                "Expected lag (min)": "Expected",
                "Excess lag (min)": "Excess",
            }
        )
    )
    st.dataframe(compact_frame, width="stretch", hide_index=True)
    st.caption(
        "Expected lag captures natural pipeline delay; excess lag shows behind-schedule time for pipeline stages. Source freshness uses raw source age."
    )
    st.caption("Open the row details below for the schedule rule, operator meaning, and artifact path summary.")

    for row in display_frame.to_dict("records"):
        expander_label = f"{row['Data source']} - {row['Status']} - {row['Last updated (UTC)']}"
        with st.expander(expander_label):
            st.markdown(f"**Schedule rule:** {row['Expected cadence / SLA']}")
            st.markdown(f"**Operator meaning:** {row['Operator meaning']}")
            if row["Details"] != "-":
                st.markdown(f"**Artifact / loader detail:** `{row['Details']}`")

