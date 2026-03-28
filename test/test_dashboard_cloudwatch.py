import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP_DIR = os.path.join(_REPO_ROOT, "app")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from src.monitoring.metrics.metrics_helper import build_metric_dimensions  # noqa: E402

from dashboard.cloudwatch import build_dashboard_metric_dimensions  # noqa: E402


def test_build_dashboard_metric_dimensions_matches_metric_publish_contract():
    dashboard_dimensions = build_dashboard_metric_dimensions(
        environment="production",
        endpoint_name="bikeshare-bikes-prod",
        city="paris",
        target_name="bikes",
    )

    publish_dimensions = {
        item["Name"]: item["Value"]
        for item in build_metric_dimensions(
            environment="production",
            endpoint="bikeshare-bikes-prod",
            city="paris",
            target_name="bikes",
        )
    }

    assert dashboard_dimensions == publish_dimensions
