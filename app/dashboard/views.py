"""Dashboard rendering functions.

Each function is responsible for one visual section of the dashboard.
All functions are tab-agnostic — the caller decides which tab they live in.
Design principles:
  - Non-technical language: no ML jargon visible to end users
  - Color = universal signal: red / orange / green for risk and freshness
  - Consistent plotly theme throughout
"""
from __future__ import annotations

from datetime import datetime, timezone

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from .targeting import DashboardTargetConfig

# ── Colour palette ──────────────────────────────────────────────────
_CLR_HIGH   = "#e63946"   # red      — high risk / stale
_CLR_MED    = "#f4a261"   # orange   — medium risk / delayed
_CLR_LOW    = "#2a9d8f"   # teal     — low risk / fresh
_CLR_LINE   = "#264653"   # dark blue-grey — chart lines
_CLR_THRESH = "#e63946"   # red dashed — alert threshold line

# Map from CloudWatch/internal metric names → plain-English labels
_METRIC_LABELS: dict[str, str] = {
    "PR-AUC-24h":            "Prediction Accuracy (24 h)",
    "F1-24h":                "Precision / Recall Balance (24 h)",
    "PredictionHeartbeat":   "Prediction Runs (24 h)",
    "ModelLatency":          "Endpoint Response Time (ms)",
    "Invocation5XXErrors":   "Endpoint Errors (24 h)",
}

_ENV_BADGES: dict[str, str] = {
    "production": "🟢 PRODUCTION",
    "staging":    "🟠 STAGING",
    "local":      "⚪ LOCAL",
}


# ── Status cards ────────────────────────────────────────────────────

def render_status_cards(
    *,
    target: DashboardTargetConfig,
    environment: str,
    model_version: str,
    threshold: float,
) -> None:
    badge_text = _ENV_BADGES.get(environment.lower(), f"🔵 {environment.upper()}")
    badge_color = {
        "production": "#2a9d8f",
        "staging":    "#f4a261",
        "local":      "#888888",
    }.get(environment.lower(), "#264653")

    st.markdown(
        f"""<div style="display:inline-block;padding:4px 14px;border-radius:20px;
        background:{badge_color};color:white;font-weight:600;font-size:0.85rem;
        margin-bottom:0.6rem">{badge_text}</div>""",
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    cols[0].metric("Prediction target", target.display_name)
    cols[1].metric("Active endpoint", target.endpoint_name)
    cols[2].metric("Model version", model_version or "unknown")
    cols[3].metric("Alert threshold", f"{threshold:.2f}")
    st.caption(
        f"City: paris &nbsp;|&nbsp; Environment: {environment} "
        f"&nbsp;|&nbsp; Label: {target.label_column} "
        f"&nbsp;|&nbsp; Score: {target.score_column}"
    )


# ── Alert summary banner ─────────────────────────────────────────────

def render_alert_banner(*, predictions: pd.DataFrame, target: DashboardTargetConfig) -> None:
    if predictions.empty:
        st.info(f"No predictions loaded yet for **{target.display_name}**.")
        return
    n_high   = int((predictions["score"] >= 0.7).sum())
    n_medium = int(((predictions["score"] >= 0.4) & (predictions["score"] < 0.7)).sum())
    n_total  = len(predictions)

    if n_high > 0:
        st.error(
            f"⚠️ **{n_high} station{'s' if n_high != 1 else ''} at HIGH risk** of running out of "
            f"{'bikes' if target.target_name == 'bikes' else 'docks'} "
            f"in the next 30 minutes &nbsp;|&nbsp; "
            f"{n_medium} at medium risk &nbsp;|&nbsp; {n_total} stations total"
        )
    elif n_medium > 0:
        st.warning(
            f"🟡 **{n_medium} station{'s' if n_medium != 1 else ''} at MEDIUM risk** &nbsp;|&nbsp; "
            f"No high-risk stations currently &nbsp;|&nbsp; {n_total} stations total"
        )
    else:
        st.success(
            f"✅ **All {n_total} stations look healthy** — no high or medium risk detected right now"
        )


# ── Prediction map ───────────────────────────────────────────────────

def render_prediction_map(
    *,
    station_info: pd.DataFrame,
    predictions: pd.DataFrame,
    target: DashboardTargetConfig,
) -> str | None:
    st.subheader(f"Live map — {target.display_name} risk")
    if station_info.empty or predictions.empty:
        st.info("Map data is not available yet.")
        return None

    merged = station_info.merge(predictions, on="station_id", how="inner")
    if merged.empty:
        st.info("No station rows overlap with the latest predictions.")
        return None

    merged["score"] = pd.to_numeric(merged["score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    center = [merged["lat"].mean(), merged["lon"].mean()]
    fmap = folium.Map(location=center, zoom_start=12, tiles="CartoDB Positron")

    for row in merged.itertuples():
        if row.score >= 0.7:
            color, risk_label = _CLR_HIGH, "High risk"
        elif row.score >= 0.4:
            color, risk_label = _CLR_MED, "Medium risk"
        else:
            color, risk_label = _CLR_LOW, "Low risk"

        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=f"{row.name} — {risk_label} ({row.score:.2f})",
            popup=folium.Popup(
                f"<b>{row.name}</b><br>{target.display_name}: {row.score:.3f}<br>{risk_label}",
                max_width=220,
            ),
        ).add_to(fmap)

    event = st_folium(fmap, width=None, height=440, returned_objects=["last_object_clicked_tooltip"])

    # Legend
    st.markdown(
        f"<span style='color:{_CLR_HIGH}'>⬤</span>&nbsp;High risk (≥ 0.7)&emsp;"
        f"<span style='color:{_CLR_MED}'>⬤</span>&nbsp;Medium (0.4 – 0.7)&emsp;"
        f"<span style='color:{_CLR_LOW}'>⬤</span>&nbsp;Low (< 0.4)",
        unsafe_allow_html=True,
    )

    clicked = event.get("last_object_clicked_tooltip") if isinstance(event, dict) else None
    if clicked and " — " in clicked:
        station_name = clicked.split(" — ")[0]
        match = merged[merged["name"] == station_name]
        if not match.empty:
            return str(match.iloc[0]["station_id"])
    return None


# ── Top-N risk table ─────────────────────────────────────────────────

def render_top_risk_table(
    *,
    predictions: pd.DataFrame,
    target: DashboardTargetConfig,
    top_n: int,
) -> None:
    st.subheader(f"Top {top_n} highest-risk stations")
    if predictions.empty:
        st.info("No predictions available.")
        return

    ranked = predictions.sort_values("score", ascending=False).head(top_n).copy()
    ranked["Risk level"] = ranked["score"].apply(
        lambda s: "🔴 High" if s >= 0.7 else ("🟡 Medium" if s >= 0.4 else "🟢 Low")
    )
    ranked = ranked.rename(columns={"station_id": "Station ID", "ts": "Last updated", "score": "Risk score"})

    st.dataframe(
        ranked[["Station ID", "Last updated", "Risk level", "Risk score"]],
        use_container_width=True,
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


# ── Prediction history chart ─────────────────────────────────────────

def render_history_chart(
    *,
    history: pd.DataFrame,
    target: DashboardTargetConfig,
    threshold: float = 0.5,
) -> None:
    st.subheader("Risk score history — selected station")
    if history.empty:
        st.info("Select a station on the map to see its history, or wait for data to load.")
        return

    df = history.sort_values("ts")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["ts"],
        y=df["score"],
        mode="lines+markers",
        name="Risk score",
        line=dict(color=_CLR_LINE, width=2),
        marker=dict(size=5),
    ))
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=_CLR_THRESH,
        annotation_text=f"Alert threshold ({threshold:.2f})",
        annotation_position="bottom right",
    )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Stockout risk score (0 = safe, 1 = high risk)",
        yaxis=dict(range=[0, 1.05]),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Model / System health metrics ────────────────────────────────────

def render_metric_section(
    *,
    title: str,
    series_map: dict[str, pd.DataFrame],
    human_labels: dict[str, str] | None = None,
) -> None:
    """Render a row of metric cards with sparkline charts.

    human_labels: optional mapping from internal metric name → display name.
    Falls back to _METRIC_LABELS global dict, then to the raw name.
    """
    st.subheader(title)
    labels = {**_METRIC_LABELS, **(human_labels or {})}
    cols = st.columns(max(1, len(series_map)))

    for idx, (metric_name, series) in enumerate(series_map.items()):
        display_name = labels.get(metric_name, metric_name)
        with cols[idx]:
            if series.empty or metric_name not in series.columns:
                st.info(f"{display_name}: no data")
                continue

            current = float(series[metric_name].iloc[-1])
            prev    = float(series[metric_name].iloc[-2]) if len(series) > 1 else current
            delta   = current - prev

            st.metric(
                label=display_name,
                value=f"{current:.3f}" if current < 1000 else f"{current:.0f}",
                delta=f"{delta:+.3f}" if current < 1000 else f"{delta:+.0f}",
            )

            fig = go.Figure(go.Scatter(
                x=series["ts"],
                y=series[metric_name],
                mode="lines",
                line=dict(color=_CLR_LINE, width=2),
                fill="tozeroy",
                fillcolor="rgba(38,70,83,0.08)",
            ))
            fig.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
                showlegend=False,
                height=160,
            )
            st.plotly_chart(fig, use_container_width=True)


# ── Data freshness table ─────────────────────────────────────────────

def render_freshness_table(*, freshness: pd.DataFrame) -> None:
    st.subheader("Data pipeline status")
    if freshness.empty:
        st.info("No freshness data available.")
        return

    now_utc = datetime.now(timezone.utc)
    frame = freshness.copy()
    frame = frame.rename(columns={"source": "Data source"})

    if "latest_dt_str" in frame.columns:
        frame["Last updated (UTC)"] = pd.to_datetime(
            frame["latest_dt_str"], format="%Y-%m-%d-%H-%M", errors="coerce", utc=True
        )
        delay_series = (now_utc - frame["Last updated (UTC)"]).dt.total_seconds() / 60
        frame["Delay (min)"] = delay_series.round(1)
        frame["Status"] = frame["Delay (min)"].apply(
            lambda d: "🟢 Fresh" if d < 10 else ("🟡 Delayed" if d < 60 else "🔴 Stale")
            if pd.notna(d) else "⚪ Unknown"
        )

    display_cols = ["Data source", "Last updated (UTC)", "Delay (min)", "Status"]
    display_cols = [c for c in display_cols if c in frame.columns]
    st.dataframe(
        frame[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Delay (min)": st.column_config.NumberColumn(format="%.1f min"),
        },
    )
