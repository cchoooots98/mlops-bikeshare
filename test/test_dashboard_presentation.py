import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP_DIR = os.path.join(_REPO_ROOT, "app")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from dashboard.contracts import ArtifactLoadResult, FreshnessLoadResult, LoadStatus  # noqa: E402
from dashboard.presentation import (  # noqa: E402
    MetricSpec,
    build_data_status_frame,
    build_station_risk_frame,
    classify_metric_status,
    resolve_selected_station,
    station_history_context,
    station_history_title,
    summarize_quality_availability,
)


def test_build_station_risk_frame_adds_business_columns_and_sorts_by_risk():
    station_info = pd.DataFrame(
        [
            {
                "station_id": "2",
                "station_name": "Bravo",
                "dt": "2026-03-19-00-35",
                "bikes": 1,
                "docks": 12,
                "capacity": 20,
                "lat": 48.85,
                "lon": 2.35,
                "util_bikes": 0.05,
                "util_docks": 0.60,
            },
            {
                "station_id": "1",
                "station_name": "Alpha",
                "dt": "2026-03-19-00-35",
                "bikes": 10,
                "docks": 8,
                "capacity": 20,
                "lat": 48.86,
                "lon": 2.36,
                "util_bikes": 0.50,
                "util_docks": 0.40,
            },
        ]
    )
    prediction_result = ArtifactLoadResult(
        status=LoadStatus.OK,
        data=pd.DataFrame(
            [
                {"station_id": "1", "ts": datetime(2026, 3, 19, 0, 35, tzinfo=timezone.utc), "score": 0.42},
                {"station_id": "2", "ts": datetime(2026, 3, 19, 0, 35, tzinfo=timezone.utc), "score": 0.91},
            ]
        ),
    )

    frame = build_station_risk_frame(
        station_info=station_info,
        prediction_result=prediction_result,
        target_name="bikes",
        threshold=0.60,
    )

    assert list(frame["station_id"]) == ["2", "1"]
    assert set(
        [
            "station_name",
            "bikes",
            "docks",
            "capacity",
            "risk_level",
            "stockout_probability",
            "current_status",
        ]
    ).issubset(frame.columns)
    assert frame.iloc[0]["risk_level"] == "Critical"
    assert frame.iloc[0]["current_status"] == "Stockout now"


def test_build_station_risk_frame_breaks_ties_by_inventory_then_capacity():
    station_info = pd.DataFrame(
        [
            {
                "station_id": "1",
                "station_name": "Alpha",
                "dt": "2026-03-19-00-35",
                "bikes": 7,
                "docks": 13,
                "capacity": 20,
                "lat": 48.86,
                "lon": 2.36,
            },
            {
                "station_id": "2",
                "station_name": "Bravo",
                "dt": "2026-03-19-00-35",
                "bikes": 7,
                "docks": 12,
                "capacity": 24,
                "lat": 48.85,
                "lon": 2.35,
            },
            {
                "station_id": "3",
                "station_name": "Charlie",
                "dt": "2026-03-19-00-35",
                "bikes": 9,
                "docks": 10,
                "capacity": 18,
                "lat": 48.84,
                "lon": 2.34,
            },
        ]
    )
    prediction_result = ArtifactLoadResult(
        status=LoadStatus.OK,
        data=pd.DataFrame(
            [
                {"station_id": "1", "ts": datetime(2026, 3, 19, 0, 35, tzinfo=timezone.utc), "score": 0.70},
                {"station_id": "2", "ts": datetime(2026, 3, 19, 0, 35, tzinfo=timezone.utc), "score": 0.70},
                {"station_id": "3", "ts": datetime(2026, 3, 19, 0, 35, tzinfo=timezone.utc), "score": 0.70},
            ]
        ),
    )

    frame = build_station_risk_frame(
        station_info=station_info,
        prediction_result=prediction_result,
        target_name="bikes",
        threshold=0.60,
    )

    assert list(frame["station_id"]) == ["2", "1", "3"]


def test_resolve_selected_station_falls_back_to_highest_risk_station():
    frame = pd.DataFrame(
        [
            {"station_id": "10", "station_name": "Top Risk", "score": 0.95},
            {"station_id": "11", "station_name": "Lower Risk", "score": 0.50},
        ]
    )

    selected = resolve_selected_station(station_risk_frame=frame, selected_station_id=None)

    assert selected is not None
    assert selected["station_id"] == "10"


def test_station_history_title_is_clean_section_label():
    title = station_history_title()

    assert title == "Station History"


def test_station_history_context_uses_station_name_and_id():
    context = station_history_context({"station_id": "123", "station_name": "Republique"})

    assert context == "Station: Republique (123). Forecast horizon: next 30 minutes (UTC)."


def test_summarize_quality_availability_explains_missing_cloudwatch_metrics():
    quality_result = ArtifactLoadResult(
        status=LoadStatus.OK,
        latest_dt=datetime.now(timezone.utc) - timedelta(minutes=5),
    )
    metric_series_map = {
        "PR-AUC-24h": pd.DataFrame(columns=["ts", "PR-AUC-24h"]),
        "F1-24h": pd.DataFrame(columns=["ts", "F1-24h"]),
    }

    severity, title, message = summarize_quality_availability(
        quality_result=quality_result,
        metric_series_map=metric_series_map,
    )

    assert severity == "info"
    assert "CloudWatch quality metrics" in title
    assert "CloudWatch series" in message


def test_summarize_quality_availability_flags_stale_quality_artifact():
    quality_result = ArtifactLoadResult(
        status=LoadStatus.OK,
        latest_dt=datetime.now(timezone.utc) - timedelta(days=2),
    )

    severity, title, message = summarize_quality_availability(
        quality_result=quality_result,
        metric_series_map={"PR-AUC-24h": pd.DataFrame(columns=["ts", "PR-AUC-24h"])},
    )

    assert severity == "warning"
    assert "stale" in title.lower()
    assert "older than 24 hours" in message


def test_build_data_status_frame_marks_stale_and_waiting_sources():
    now = datetime(2026, 3, 27, 15, 0, tzinfo=timezone.utc)
    prediction_result = ArtifactLoadResult(
        status=LoadStatus.OK,
        latest_dt=now - timedelta(minutes=61),
        latest_key="predictions/latest",
        source_name="Prediction artifact",
    )
    quality_result = ArtifactLoadResult(
        status=LoadStatus.NO_OBJECTS,
        latest_dt=None,
        source_name="Quality artifact",
    )
    freshness_result = FreshnessLoadResult(
        status=LoadStatus.OK,
        data=pd.DataFrame(
            [
                {
                    "source": "Source freshness",
                    "freshness_type": "source",
                    "latest_dt_str": "2026-03-27-14-55",
                    "loader_status": "ok",
                    "message": "",
                },
                {
                    "source": "Feature freshness",
                    "freshness_type": "feature",
                    "latest_dt_str": "2026-03-27-14-55",
                    "loader_status": "ok",
                    "message": "",
                },
            ]
        ),
    )

    frame = build_data_status_frame(
        prediction_result=prediction_result,
        quality_result=quality_result,
        freshness_result=freshness_result,
        prediction_sla_minutes=30,
        quality_sla_minutes=45,
        feature_sla_minutes=60,
        now_utc=now,
    )

    rows = {row["Data source"]: row for row in frame.to_dict("records")}
    assert rows["Prediction artifact"]["Status"] == "Critical"
    assert rows["Quality artifact"]["Status"] == "Critical"
    assert rows["Prediction artifact"]["Expected lag (min)"] == 20.0
    assert rows["Prediction artifact"]["Excess lag (min)"] == 41.0
    assert "natural lag 43-58 min" in rows["Quality artifact"]["Expected cadence / SLA"]
    assert rows["Source freshness"]["Status"] == "Healthy"
    assert rows["Feature freshness"]["Status"] == "Healthy"


def test_build_data_status_frame_caps_downstream_stages_to_stale_source():
    now = datetime(2026, 3, 28, 0, 1, tzinfo=timezone.utc)
    prediction_result = ArtifactLoadResult(
        status=LoadStatus.OK,
        latest_dt=datetime(2026, 3, 27, 23, 50, tzinfo=timezone.utc),
        latest_key="predictions/dt=2026-03-27-23-50/predictions.parquet",
        source_name="Prediction artifact",
    )
    quality_result = ArtifactLoadResult(
        status=LoadStatus.NO_OBJECTS,
        latest_dt=None,
        source_name="Quality artifact",
    )
    freshness_result = FreshnessLoadResult(
        status=LoadStatus.OK,
        data=pd.DataFrame(
            [
                {
                    "source": "Source freshness",
                    "freshness_type": "source",
                    "latest_dt_str": datetime(2026, 3, 27, 23, 50, tzinfo=timezone.utc),
                    "loader_status": "ok",
                    "message": "",
                },
                {
                    "source": "Feature freshness",
                    "freshness_type": "feature",
                    "latest_dt_str": "2026-03-27-23-50",
                    "loader_status": "ok",
                    "message": "",
                },
            ]
        ),
    )

    frame = build_data_status_frame(
        prediction_result=prediction_result,
        quality_result=quality_result,
        freshness_result=freshness_result,
        now_utc=now,
    )

    rows = {row["Data source"]: row for row in frame.to_dict("records")}
    assert rows["Source freshness"]["Status"] == "Warning"
    assert rows["Feature freshness"]["Status"] == "Healthy"
    assert rows["Feature freshness"]["Expected lag (min)"] == 11.0
    assert rows["Feature freshness"]["Excess lag (min)"] == 0.0
    assert rows["Prediction artifact"]["Status"] == "Healthy"
    assert rows["Prediction artifact"]["Expected lag (min)"] == 11.0
    assert rows["Prediction artifact"]["Excess lag (min)"] == 0.0
    assert "capped by latest source freshness" in rows["Feature freshness"]["Operator meaning"].lower()
    assert rows["Quality artifact"]["Status"] == "Healthy"
    assert "No quality artifact is expected yet" in rows["Quality artifact"]["Operator meaning"]


def test_build_data_status_frame_treats_natural_quality_delay_as_healthy():
    now = datetime(2026, 3, 27, 15, 0, tzinfo=timezone.utc)
    prediction_result = ArtifactLoadResult(
        status=LoadStatus.OK,
        latest_dt=datetime(2026, 3, 27, 14, 40, tzinfo=timezone.utc),
        latest_key="predictions/dt=2026-03-27-14-40/predictions.parquet",
        source_name="Prediction artifact",
    )
    quality_result = ArtifactLoadResult(
        status=LoadStatus.OK,
        latest_dt=datetime(2026, 3, 27, 14, 10, tzinfo=timezone.utc),
        latest_key="quality/dt=2026-03-27-14-10/quality.parquet",
        source_name="Quality artifact",
    )
    freshness_result = FreshnessLoadResult(
        status=LoadStatus.OK,
        data=pd.DataFrame(
            [
                {
                    "source": "feat_station_snapshot_latest",
                    "latest_dt_str": "2026-03-27-14-55",
                    "loader_status": "ok",
                    "message": "",
                }
            ]
        ),
    )

    frame = build_data_status_frame(
        prediction_result=prediction_result,
        quality_result=quality_result,
        freshness_result=freshness_result,
        now_utc=now,
    )

    rows = {row["Data source"]: row for row in frame.to_dict("records")}
    assert rows["Prediction artifact"]["Status"] == "Healthy"
    assert rows["Prediction artifact"]["Delay (min)"] == 20.0
    assert rows["Prediction artifact"]["Expected lag (min)"] == 20.0
    assert rows["Prediction artifact"]["Excess lag (min)"] == 0.0
    assert rows["Quality artifact"]["Status"] == "Healthy"
    assert rows["Quality artifact"]["Delay (min)"] == 50.0
    assert rows["Quality artifact"]["Expected lag (min)"] == 50.0
    assert rows["Quality artifact"]["Excess lag (min)"] == 0.0
    assert rows["feat_station_snapshot_latest"]["Status"] == "Healthy"


def test_build_data_status_frame_separates_source_freshness_from_feature_schedule():
    now = datetime(2026, 3, 27, 15, 0, tzinfo=timezone.utc)
    prediction_result = ArtifactLoadResult(
        status=LoadStatus.OK,
        latest_dt=datetime(2026, 3, 27, 14, 40, tzinfo=timezone.utc),
    )
    quality_result = ArtifactLoadResult(
        status=LoadStatus.OK,
        latest_dt=datetime(2026, 3, 27, 14, 10, tzinfo=timezone.utc),
    )
    freshness_result = FreshnessLoadResult(
        status=LoadStatus.OK,
        data=pd.DataFrame(
            [
                {
                    "source": "Source freshness",
                    "freshness_type": "source",
                    "latest_dt_str": datetime(2026, 3, 27, 14, 47, tzinfo=timezone.utc),
                    "loader_status": "ok",
                    "message": "",
                },
                {
                    "source": "Feature freshness",
                    "freshness_type": "feature",
                    "latest_dt_str": "2026-03-27-14-55",
                    "loader_status": "ok",
                    "message": "",
                },
            ]
        ),
    )

    frame = build_data_status_frame(
        prediction_result=prediction_result,
        quality_result=quality_result,
        freshness_result=freshness_result,
        now_utc=now,
    )

    rows = {row["Data source"]: row for row in frame.to_dict("records")}
    assert rows["Source freshness"]["Status"] == "Warning"
    assert pd.isna(rows["Source freshness"]["Expected lag (min)"])
    assert pd.isna(rows["Source freshness"]["Excess lag (min)"])
    assert "10+ min" in rows["Source freshness"]["Expected cadence / SLA"]
    assert rows["Feature freshness"]["Status"] == "Healthy"


def test_build_data_status_frame_caps_quality_expectation_to_latest_prediction_dt():
    now = datetime(2026, 3, 28, 0, 56, tzinfo=timezone.utc)
    prediction_result = ArtifactLoadResult(
        status=LoadStatus.OK,
        latest_dt=datetime(2026, 3, 27, 23, 50, tzinfo=timezone.utc),
        latest_key="predictions/dt=2026-03-27-23-50/predictions.parquet",
    )
    quality_result = ArtifactLoadResult(
        status=LoadStatus.OK,
        latest_dt=datetime(2026, 3, 27, 23, 50, tzinfo=timezone.utc),
        latest_key="quality/dt=2026-03-27-23-50/quality.parquet",
    )
    freshness_result = FreshnessLoadResult(
        status=LoadStatus.OK,
        data=pd.DataFrame(
            [
                {
                    "source": "Source freshness",
                    "freshness_type": "source",
                    "latest_dt_str": datetime(2026, 3, 27, 23, 50, tzinfo=timezone.utc),
                    "loader_status": "ok",
                    "message": "",
                },
                {
                    "source": "Feature freshness",
                    "freshness_type": "feature",
                    "latest_dt_str": "2026-03-27-23-50",
                    "loader_status": "ok",
                    "message": "",
                },
            ]
        ),
    )

    frame = build_data_status_frame(
        prediction_result=prediction_result,
        quality_result=quality_result,
        freshness_result=freshness_result,
        now_utc=now,
    )

    rows = {row["Data source"]: row for row in frame.to_dict("records")}
    assert rows["Prediction artifact"]["Status"] == "Healthy"
    assert rows["Prediction artifact"]["Expected lag (min)"] == 66.0
    assert rows["Prediction artifact"]["Excess lag (min)"] == 0.0
    assert rows["Quality artifact"]["Status"] == "Healthy"
    assert rows["Quality artifact"]["Delay (min)"] == 66.0
    assert rows["Quality artifact"]["Expected lag (min)"] == 66.0
    assert rows["Quality artifact"]["Excess lag (min)"] == 0.0
    assert "capped by latest prediction artifact" in rows["Quality artifact"]["Operator meaning"].lower()


def test_classify_metric_status_respects_sla_bands():
    pr_auc_spec = MetricSpec(label="PR-AUC (24h)", direction="higher", warning=0.70, critical=0.55)
    latency_spec = MetricSpec(label="Latency", direction="lower", warning=200.0, critical=300.0, decimals=0)
    observed_spec = MetricSpec(label="Threshold Hit Rate", direction="none")

    assert classify_metric_status(0.80, pr_auc_spec)[0] == "Healthy"
    assert classify_metric_status(0.60, pr_auc_spec)[0] == "Warning"
    assert classify_metric_status(0.40, pr_auc_spec)[0] == "Critical"
    assert classify_metric_status(180.0, latency_spec)[0] == "Healthy"
    assert classify_metric_status(250.0, latency_spec)[0] == "Warning"
    assert classify_metric_status(450.0, latency_spec)[0] == "Critical"
    assert classify_metric_status(0.12, observed_spec)[0] == "Observed"
