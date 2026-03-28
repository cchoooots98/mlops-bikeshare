from datetime import datetime, timedelta, timezone

from airflow.dags.external_task_utils import resolve_latest_upstream_logical_date


def test_daily_schedule_mapping_uses_project_timezone_not_utc():
    upstream_logical_date = resolve_latest_upstream_logical_date(
        datetime(2026, 3, 27, 1, 22, tzinfo=timezone.utc),
        upstream_schedule="22 1 * * *",
        data_interval_end=datetime(2026, 3, 28, 1, 22, tzinfo=timezone.utc),
        minimum_age=timedelta(),
    )

    assert upstream_logical_date == datetime(2026, 3, 27, 0, 22, tzinfo=timezone.utc)


def test_quality_schedule_mapping_respects_30_minute_maturity_window():
    upstream_logical_date = resolve_latest_upstream_logical_date(
        datetime(2026, 3, 28, 1, 3, tzinfo=timezone.utc),
        upstream_schedule="2-59/15 * * * *",
        data_interval_end=datetime(2026, 3, 28, 1, 18, tzinfo=timezone.utc),
        minimum_age=timedelta(minutes=30),
    )

    assert upstream_logical_date == datetime(2026, 3, 28, 0, 32, tzinfo=timezone.utc)
