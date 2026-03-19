import os
import sys
from contextlib import nullcontext

import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP_DIR = os.path.join(_REPO_ROOT, "app")
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from dashboard.queries import load_station_info  # noqa: E402


class _FakeEngine:
    def connect(self):
        return nullcontext(object())


def test_load_station_info_returns_business_ready_station_columns(monkeypatch):
    expected = pd.DataFrame(
        [
            {
                "station_id": "123",
                "station_name": "Republique",
                "dt": "2026-03-19-00-35",
                "bikes": 5,
                "docks": 10,
                "capacity": 20,
                "lat": 48.86,
                "lon": 2.36,
                "util_bikes": 0.25,
                "util_docks": 0.50,
            }
        ]
    )
    captured = {}

    def _fake_read_sql(sql, conn, params):  # noqa: ANN001
        captured["sql"] = str(sql)
        captured["params"] = params
        return expected

    monkeypatch.setattr(pd, "read_sql", _fake_read_sql)

    result = load_station_info(engine=_FakeEngine(), schema="analytics", city="paris")

    assert result.equals(expected)
    assert captured["params"] == {"city": "paris"}
    assert "AS station_name" in captured["sql"]
    assert "f.dt" in captured["sql"]
    assert "f.bikes" in captured["sql"]
    assert "f.docks" in captured["sql"]
