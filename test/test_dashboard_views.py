import os
import sys
from datetime import datetime, timezone

import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP_DIR = os.path.join(_REPO_ROOT, "app")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from dashboard import views  # noqa: E402
from dashboard.contracts import (  # noqa: E402
    ArtifactLoadResult,
    FreshnessLoadResult,
    LoadStatus,
)
from dashboard.presentation import MetricSpec  # noqa: E402
from dashboard.targeting import DashboardTargetConfig  # noqa: E402


class _DummyColumn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _target() -> DashboardTargetConfig:
    return DashboardTargetConfig(
        target_name="bikes",
        display_name="Bike stockout",
        label_column="label",
        score_column="score",
        score_bin_column="score_bin",
        endpoint_name="bikeshare-bikes-staging",
        inference_prefix="predictions",
        quality_prefix="quality",
        section_title="Bike stockout risk",
    )


def test_render_metric_section_uses_unique_plotly_keys(monkeypatch):
    captured_keys = []

    monkeypatch.setattr(views.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(views.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(views.st, "columns", lambda count: [_DummyColumn() for _ in range(count)])
    monkeypatch.setattr(views.st, "metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(views.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(views, "_render_status_chip", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        views.st,
        "plotly_chart",
        lambda fig, **kwargs: captured_keys.append(kwargs.get("key")),
    )

    series_map = {
        "PR-AUC-24h": pd.DataFrame(
            {
                "ts": [
                    datetime(2026, 3, 19, 10, 0, tzinfo=timezone.utc),
                    datetime(2026, 3, 19, 11, 0, tzinfo=timezone.utc),
                ],
                "PR-AUC-24h": [0.61, 0.64],
            }
        ),
        "F1-24h": pd.DataFrame(
            {
                "ts": [
                    datetime(2026, 3, 19, 10, 0, tzinfo=timezone.utc),
                    datetime(2026, 3, 19, 11, 0, tzinfo=timezone.utc),
                ],
                "F1-24h": [0.40, 0.45],
            }
        ),
    }

    views.render_metric_section(
        title="Prediction Quality",
        description="Last 24 hours in UTC.",
        key_prefix="prediction-quality",
        series_map=series_map,
        metric_specs={
            "PR-AUC-24h": MetricSpec(label="PR-AUC (24h)", direction="higher", warning=0.70, critical=0.55),
            "F1-24h": MetricSpec(label="F1 (24h)", direction="higher", warning=0.55, critical=0.40),
        },
    )

    assert captured_keys == [
        "metric-chart-prediction-quality-pr-auc-24h",
        "metric-chart-prediction-quality-f1-24h",
    ]


def test_render_metric_section_sums_counter_style_metrics_over_visible_window(
    monkeypatch,
):
    captured_metrics = []

    monkeypatch.setattr(views.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(views.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(views.st, "columns", lambda count: [_DummyColumn() for _ in range(count)])
    monkeypatch.setattr(views.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(views, "_render_status_chip", lambda *args, **kwargs: None)
    monkeypatch.setattr(views.st, "metric", lambda **kwargs: captured_metrics.append(kwargs))
    monkeypatch.setattr(views.st, "plotly_chart", lambda *args, **kwargs: None)

    views.render_metric_section(
        title="System Health",
        series_map={
            "Invocations": pd.DataFrame(
                {
                    "ts": [
                        datetime(2026, 3, 25, 10, 0, tzinfo=timezone.utc),
                        datetime(2026, 3, 25, 10, 5, tzinfo=timezone.utc),
                        datetime(2026, 3, 25, 10, 10, tzinfo=timezone.utc),
                    ],
                    "Invocations": [120.0, 0.0, 80.0],
                }
            )
        },
        metric_specs={
            "Invocations": MetricSpec(
                label="Invocations (24h total)",
                direction="higher",
                decimals=0,
                summary="window_sum",
            )
        },
    )

    assert captured_metrics == [{"label": "Invocations (24h total)", "value": "200"}]


def test_render_history_chart_uses_unique_key_and_clean_text(monkeypatch):
    captured = {"subheader": [], "caption": [], "key": None}

    monkeypatch.setattr(views.st, "subheader", lambda text: captured["subheader"].append(text))
    monkeypatch.setattr(views.st, "caption", lambda text: captured["caption"].append(text))
    monkeypatch.setattr(
        views.st,
        "plotly_chart",
        lambda fig, **kwargs: captured.update({"key": kwargs.get("key")}),
    )

    history_result = ArtifactLoadResult(
        status=LoadStatus.OK,
        data=pd.DataFrame(
            {
                "ts": [
                    datetime(2026, 3, 19, 10, 0, tzinfo=timezone.utc),
                    datetime(2026, 3, 19, 11, 0, tzinfo=timezone.utc),
                ],
                "score": [0.20, 0.65],
            }
        ),
    )

    views.render_history_chart(
        history_result=history_result,
        target=_target(),
        threshold=0.60,
        selected_station={
            "station_id": "54000604",
            "station_name": "Ordener - Poissonniers",
        },
    )

    assert captured["subheader"] == ["Station History"]
    assert (
        captured["caption"][0] == "Station: Ordener - Poissonniers (54000604). Forecast horizon: next 30 minutes (UTC)."
    )
    assert captured["key"] == "history-chart-bikes-54000604"


def test_render_selected_station_summary_renders_full_status_text(monkeypatch):
    captured_markdown = []
    captured_caption = []

    monkeypatch.setattr(views.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(views.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        views.st,
        "markdown",
        lambda text, **kwargs: captured_markdown.append(text),
    )
    monkeypatch.setattr(
        views.st,
        "caption",
        lambda text: captured_caption.append(text),
    )

    views.render_selected_station_summary(
        selected_station={
            "station_id": "54000604",
            "station_name": "Gergovie - Vercingetorix",
            "bikes": 4,
            "docks": 25,
            "capacity": 29,
            "current_status": "Low inventory now",
            "risk_level": "Normal",
            "stockout_probability": 0.154,
            "ts": datetime(2026, 3, 19, 11, 50, tzinfo=timezone.utc),
        },
        target=_target(),
    )

    assert "Gergovie - Vercingetorix" in captured_markdown[0]
    assert "Low inventory now" in captured_markdown[1]
    assert captured_caption[0] == "Station ID: 54000604. Forecast horizon: next 30 minutes (UTC)."


def test_render_data_status_table_formats_timestamps_and_compacts_details(monkeypatch):
    captured = {"frame": None, "caption": []}

    monkeypatch.setattr(views.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(views.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(views.st, "caption", lambda text: captured["caption"].append(text))
    monkeypatch.setattr(
        views.st,
        "dataframe",
        lambda frame, **kwargs: captured.update({"frame": frame.copy()}),
    )
    monkeypatch.setattr(
        views,
        "build_data_status_frame",
        lambda **kwargs: pd.DataFrame(
            [
                {
                    "Data source": "Prediction artifact",
                    "Last updated (UTC)": datetime(2026, 3, 25, 23, 15, tzinfo=timezone.utc),
                    "Delay (min)": 15.0,
                    "Status": "Healthy",
                    "Expected cadence / SLA": "Prediction every 15 min; stale after 30 min",
                    "Operator meaning": "Freshness is within the 30 minute expectation.",
                    "Details": "inference/target=bikes/city=paris/dt=2026-03-25-23-15/predictions.parquet",
                }
            ]
        ),
    )

    views.render_data_status_table(
        prediction_result=ArtifactLoadResult(status=LoadStatus.OK),
        quality_result=ArtifactLoadResult(status=LoadStatus.OK),
        freshness_result=FreshnessLoadResult(status=LoadStatus.OK, data=pd.DataFrame()),
        prediction_sla_minutes=30,
        quality_sla_minutes=45,
        feature_sla_minutes=60,
    )

    assert captured["frame"] is not None
    first_row = captured["frame"].iloc[0].to_dict()
    assert first_row["Last updated (UTC)"] == "2026-03-25 23:15 UTC"
    assert first_row["Delay (min)"] == "15.0 min"
    assert first_row["Details"] == "inference/target=bikes/city=paris/.../dt=2026-03-25-23-15/predictions.parquet"
    assert "S3 and Airflow logs" in captured["caption"][-1]


def test_render_top_risk_table_filters_zero_capacity_rows_before_ranking(monkeypatch):
    captured = {"frame": None}

    monkeypatch.setattr(views.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(views.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(views.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        views.st,
        "dataframe",
        lambda frame, **kwargs: captured.update({"frame": frame.copy()}),
    )

    views.render_top_risk_table(
        station_risk_frame=pd.DataFrame(
            [
                {
                    "station_id": "1",
                    "station_name": "Zero Capacity Highest Risk",
                    "bikes": 0,
                    "docks": 0,
                    "capacity": 0,
                    "current_status": "Stockout now",
                    "stockout_probability": 1.0,
                    "ts": datetime(2026, 3, 26, 21, 10, tzinfo=timezone.utc),
                },
                {
                    "station_id": "2",
                    "station_name": "Positive Capacity A",
                    "bikes": 0,
                    "docks": 30,
                    "capacity": 30,
                    "current_status": "Stockout now",
                    "stockout_probability": 0.98,
                    "ts": datetime(2026, 3, 26, 21, 10, tzinfo=timezone.utc),
                },
                {
                    "station_id": "3",
                    "station_name": "Positive Capacity B",
                    "bikes": 1,
                    "docks": 24,
                    "capacity": 25,
                    "current_status": "Stockout now",
                    "stockout_probability": 0.97,
                    "ts": datetime(2026, 3, 26, 21, 10, tzinfo=timezone.utc),
                },
            ]
        ),
        top_n=2,
        target=_target(),
    )

    assert captured["frame"] is not None
    assert list(captured["frame"]["Station ID"]) == ["2", "3"]
    assert list(captured["frame"]["Capacity"]) == [30, 25]


def test_render_top_risk_table_warns_when_no_positive_capacity_rows_remain(monkeypatch):
    captured = {"warning": None, "frame_called": False}

    monkeypatch.setattr(views.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(views.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        views.st,
        "warning",
        lambda message: captured.update({"warning": message}),
    )
    monkeypatch.setattr(
        views.st,
        "dataframe",
        lambda *args, **kwargs: captured.update({"frame_called": True}),
    )

    views.render_top_risk_table(
        station_risk_frame=pd.DataFrame(
            [
                {
                    "station_id": "1",
                    "station_name": "Zero Capacity A",
                    "bikes": 0,
                    "docks": 0,
                    "capacity": 0,
                    "current_status": "Stockout now",
                    "stockout_probability": 1.0,
                    "ts": datetime(2026, 3, 26, 21, 10, tzinfo=timezone.utc),
                },
                {
                    "station_id": "2",
                    "station_name": "Zero Capacity B",
                    "bikes": 0,
                    "docks": 0,
                    "capacity": 0,
                    "current_status": "Stockout now",
                    "stockout_probability": 0.95,
                    "ts": datetime(2026, 3, 26, 21, 10, tzinfo=timezone.utc),
                },
            ]
        ),
        top_n=2,
        target=_target(),
    )

    assert captured["warning"] == "No stations with positive capacity are available for the ranked table."
    assert captured["frame_called"] is False
