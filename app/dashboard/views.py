"""Dashboard rendering functions."""
from __future__ import annotations

from datetime import datetime, timezone

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from .contracts import ArtifactLoadResult, FreshnessLoadResult, LoadStatus
from .targeting import DashboardTargetConfig

_CLR_CRITICAL = "#c1121f"
_CLR_ALERT = "#f77f00"
_CLR_OK = "#2a9d8f"
_CLR_INFO = "#577590"
_CLR_LINE = "#264653"

_METRIC_LABELS: dict[str, str] = {
    "PR-AUC-24h": "Prediction Accuracy (24 h)",
    "F1-24h": "Precision / Recall Balance (24 h)",
    "PredictionHeartbeat": "Prediction Runs (24 h)",
    "ModelLatency": "Endpoint Response Time (ms)",
    "Invocation5XXErrors": "Endpoint Errors (24 h)",
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


def classify_risk(score: float, *, alert_threshold: float, critical_threshold: float | None = None) -> tuple[str, str]:
    if critical_threshold is not None and score >= critical_threshold:
        return "Critical", _CLR_CRITICAL
    if score >= alert_threshold:
        return "Alert", _CLR_ALERT
    return "Normal", _CLR_OK


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

    critical_threshold = max(threshold, 0.7)
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
    station_info: pd.DataFrame,
    prediction_result: ArtifactLoadResult,
    target: DashboardTargetConfig,
    threshold: float,
) -> str | None:
    st.subheader(f"Live map - {target.display_name}")
    if prediction_result.status != LoadStatus.OK:
        render_artifact_issue(prediction_result, default_message="Prediction map is unavailable.")
        return None
    if station_info.empty:
        st.error("Station metadata is unavailable. The dashboard cannot place stations on the map.")
        return None

    predictions = prediction_result.data
    merged = station_info.merge(predictions, on="station_id", how="inner")
    if merged.empty:
        st.warning("Station metadata and prediction rows do not overlap yet.")
        return None

    critical_threshold = max(threshold, 0.7)
    center = [merged["lat"].mean(), merged["lon"].mean()]
    fmap = folium.Map(location=center, zoom_start=12, tiles="CartoDB Positron")

    for row in merged.itertuples():
        risk_label, color = classify_risk(row.score, alert_threshold=threshold, critical_threshold=critical_threshold)
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=f"{row.name} - {risk_label} ({row.score:.2f})",
            popup=folium.Popup(
                f"<b>{row.name}</b><br>{target.display_name}: {row.score:.3f}<br>{risk_label}",
                max_width=220,
            ),
        ).add_to(fmap)

    event = st_folium(fmap, width=None, height=440, returned_objects=["last_object_clicked_tooltip"])
    st.markdown(
        (
            f"<span style='color:{_CLR_CRITICAL}'>⬤</span> Critical "
            f"<span style='color:{_CLR_ALERT};margin-left:16px'>⬤</span> Alert "
            f"<span style='color:{_CLR_OK};margin-left:16px'>⬤</span> Normal"
        ),
        unsafe_allow_html=True,
    )

    clicked = event.get("last_object_clicked_tooltip") if isinstance(event, dict) else None
    if clicked and " - " in clicked:
        station_name = clicked.split(" - ")[0]
        match = merged[merged["name"] == station_name]
        if not match.empty:
            return str(match.iloc[0]["station_id"])
    return None


def render_top_risk_table(
    *,
    prediction_result: ArtifactLoadResult,
    target: DashboardTargetConfig,
    top_n: int,
    threshold: float,
) -> None:
    st.subheader(f"Top {top_n} stations by risk")
    if prediction_result.status != LoadStatus.OK:
        render_artifact_issue(prediction_result, default_message="Risk table is unavailable.")
        return

    critical_threshold = max(threshold, 0.7)
    ranked = prediction_result.data.sort_values("score", ascending=False).head(top_n).copy()
    ranked["Alert state"] = ranked["score"].apply(
        lambda score: classify_risk(score, alert_threshold=threshold, critical_threshold=critical_threshold)[0]
    )
    ranked = ranked.rename(columns={"station_id": "Station ID", "ts": "Last updated", "score": "Risk score"})

    st.dataframe(
        ranked[["Station ID", "Last updated", "Alert state", "Risk score"]],
        width="stretch",
        column_config={
            "Risk score": st.column_config.ProgressColumn(
                label="Risk score",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            ),
        },
        hide_index=True,
    )


def render_history_chart(
    *,
    history_result: ArtifactLoadResult,
    target: DashboardTargetConfig,
    threshold: float,
) -> None:
    st.subheader("Risk score history")
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
        annotation_text=f"Alert threshold ({threshold:.2f})",
        annotation_position="bottom right",
    )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Risk score",
        yaxis=dict(range=[0, 1.05]),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")


def render_metric_section(
    *,
    title: str,
    series_map: dict[str, pd.DataFrame],
    human_labels: dict[str, str] | None = None,
) -> None:
    st.subheader(title)
    labels = {**_METRIC_LABELS, **(human_labels or {})}
    cols = st.columns(max(1, len(series_map)))

    for idx, (metric_name, series) in enumerate(series_map.items()):
        display_name = labels.get(metric_name, metric_name)
        with cols[idx]:
            if series.empty or metric_name not in series.columns:
                st.warning(f"{display_name}: no metric samples available")
                continue

            current = float(series[metric_name].iloc[-1])
            prev = float(series[metric_name].iloc[-2]) if len(series) > 1 else current
            delta = current - prev

            st.metric(
                label=display_name,
                value=f"{current:.3f}" if current < 1000 else f"{current:.0f}",
                delta=f"{delta:+.3f}" if current < 1000 else f"{delta:+.0f}",
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
) -> None:
    st.subheader("Data pipeline status")
    rows: list[dict[str, object]] = []
    now_utc = datetime.now(timezone.utc)

    def _artifact_row(result: ArtifactLoadResult) -> dict[str, object]:
        delay = None
        if result.latest_dt is not None:
            delay = round((now_utc - result.latest_dt).total_seconds() / 60, 1)
        return {
            "Data source": result.source_name,
            "Last updated (UTC)": result.latest_dt,
            "Delay (min)": delay,
            "Status": result.status.value,
            "Details": result.message or (result.latest_key.rsplit("/", 2)[-2] if result.latest_key else ""),
        }

    rows.append(_artifact_row(prediction_result))
    rows.append(_artifact_row(quality_result))

    freshness = freshness_result.data.copy()
    if not freshness.empty:
        freshness["Last updated (UTC)"] = pd.to_datetime(
            freshness["latest_dt_str"], format="%Y-%m-%d-%H-%M", errors="coerce", utc=True
        )
        freshness["Delay (min)"] = (
            now_utc - freshness["Last updated (UTC)"]
        ).dt.total_seconds().div(60).round(1)
        freshness["Status"] = freshness.apply(
            lambda row: row["loader_status"]
            if row["loader_status"] != "ok"
            else ("stale" if pd.notna(row["Delay (min)"]) and row["Delay (min)"] >= 60 else "ok"),
            axis=1,
        )
        freshness["Details"] = freshness["message"].fillna("")
        freshness = freshness.rename(columns={"source": "Data source"})
        rows.extend(
            freshness[["Data source", "Last updated (UTC)", "Delay (min)", "Status", "Details"]].to_dict("records")
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        st.warning("No data status rows are available.")
        return
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config={"Delay (min)": st.column_config.NumberColumn(format="%.1f min")},
    )
