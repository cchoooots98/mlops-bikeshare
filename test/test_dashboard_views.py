import os
import sys
from datetime import datetime, timezone

import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP_DIR = os.path.join(_REPO_ROOT, "app")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from dashboard.contracts import ArtifactLoadResult, LoadStatus  # noqa: E402
from dashboard.presentation import MetricSpec  # noqa: E402
from dashboard.targeting import DashboardTargetConfig  # noqa: E402
from dashboard import views  # noqa: E402


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
        selected_station={"station_id": "54000604", "station_name": "Ordener - Poissonniers"},
    )

    assert captured["subheader"] == ["Station History"]
    assert captured["caption"][0] == "Station: Ordener - Poissonniers (54000604). Forecast horizon: next 30 minutes (UTC)."
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
