import importlib.util
import sys
from pathlib import Path

import pytest


def _load_check_gate_module():
    spec = importlib.util.spec_from_file_location("check_gate_module", Path("test/check_gate.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gate_requires_explicit_environment(monkeypatch):
    check_gate = _load_check_gate_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_gate.py",
            "--endpoint",
            "bikeshare-bikes-staging",
            "--city",
            "paris",
            "--region",
            "eu-west-3",
            "--target-name",
            "bikes",
        ],
    )

    with pytest.raises(SystemExit):
        check_gate.parse_args()


def test_gate_heartbeat_threshold_matches_15_minute_cadence():
    check_gate = _load_check_gate_module()

    assert check_gate.PREDICTION_CADENCE_MINUTES == 15
    assert check_gate.EXPECTED_HEARTBEATS == 96
    assert check_gate.HEARTBEAT_MIN == 92
    assert check_gate.MAX_HEARTBEAT_GAP_MINUTES == 35
    assert check_gate.PSI_CORE_WARN == 0.20
