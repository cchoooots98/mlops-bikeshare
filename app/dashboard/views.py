from __future__ import annotations

from datetime import datetime, timezone

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from .targeting import DashboardTargetConfig


def render_status_cards(*, target: DashboardTargetConfig, environment: str, model_version: str, threshold: float) -> None:
    cols = st.columns(4)
    cols[0].metric("Current target", target.display_name)
    cols[1].metric("Endpoint", target.endpoint_name)
    cols[2].metric("Model version", model_version or "unknown")
    cols[3].metric("Threshold", f"{threshold:.2f}")
    st.caption(
        f"Environment: {environment} | Label: {target.label_column} | Score: {target.score_column}"
    )


def render_prediction_map(*, station_info: pd.DataFrame, predictions: pd.DataFrame, target: DashboardTargetConfig) -> str | None:
    st.subheader(f"1) {target.section_title} map")
    if station_info.empty or predictions.empty:
        st.info("Map data is empty for the current target.")
        return None

    merged = station_info.merge(predictions, on="station_id", how="inner")
    if merged.empty:
        st.info("No station rows overlap with the latest predictions.")
        return None

    merged["score"] = pd.to_numeric(merged["score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    center = [merged["lat"].mean(), merged["lon"].mean()]
    fmap = folium.Map(location=center, zoom_start=12, tiles="CartoDB Positron")
    for row in merged.itertuples():
        color = "#d94801" if row.score >= 0.7 else "#f16913" if row.score >= 0.4 else "#31a354"
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.85,
            popup=f"{row.name} | {target.display_name}: {row.score:.2f}",
        ).add_to(fmap)
    event = st_folium(fmap, width=None, height=420)
    return event.get("last_object_clicked_tooltip") if isinstance(event, dict) else None


def render_top_risk_table(*, predictions: pd.DataFrame, target: DashboardTargetConfig, top_n: int) -> None:
    st.subheader(f"2) Top-{top_n} {target.display_name} stations")
    if predictions.empty:
        st.info("No latest predictions available.")
        return
    ranked = predictions.sort_values("score", ascending=False).head(top_n).copy()
    ranked["score"] = ranked["score"].round(3)
    st.dataframe(ranked, use_container_width=True)


def render_history_chart(*, history: pd.DataFrame, target: DashboardTargetConfig) -> None:
    st.subheader(f"3) {target.display_name} history")
    if history.empty:
        st.info("No history available for the selected station.")
        return
    fig = px.line(
        history.sort_values("ts"),
        x="ts",
        y="score",
        labels={"ts": "Timestamp", "score": target.display_name},
        title=f"{target.display_name} history",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_metric_section(*, title: str, series_map: dict[str, pd.DataFrame]) -> None:
    st.subheader(title)
    cols = st.columns(max(1, len(series_map)))
    for index, (metric_name, series) in enumerate(series_map.items()):
        with cols[index]:
            if series.empty:
                st.info(f"{metric_name}: no data")
                continue
            value_column = metric_name
            current_value = float(series[value_column].iloc[-1])
            st.metric(metric_name, f"{current_value:.3f}" if current_value < 1000 else f"{current_value:.0f}")
            fig = px.line(series.sort_values("ts"), x="ts", y=value_column, title=metric_name)
            st.plotly_chart(fig, use_container_width=True)


def render_freshness_table(*, freshness: pd.DataFrame) -> None:
    st.subheader("5) Data freshness")
    if freshness.empty:
        st.info("No freshness rows available.")
        return
    now_utc = datetime.now(timezone.utc)
    frame = freshness.copy()
    if "latest_dt_str" in frame.columns:
        frame["latest_utc"] = pd.to_datetime(frame["latest_dt_str"], format="%Y-%m-%d-%H-%M", errors="coerce", utc=True)
        frame["delay_min"] = ((now_utc - frame["latest_utc"]).dt.total_seconds() / 60).round(1)
    st.dataframe(frame, use_container_width=True)
