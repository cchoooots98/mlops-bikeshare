from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_app_module(tmp_path):
    module_path = Path("docker") / "app.py"
    spec = spec_from_file_location(f"docker_app_test_{tmp_path.name}", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ping_unhealthy_when_model_missing(tmp_path):
    module = _load_app_module(tmp_path)
    module.MODEL_DIR = str(tmp_path / "missing-model")
    module._cached_model = None
    client = module.app.test_client()

    response = client.get("/ping")

    assert response.status_code == 503


def test_ping_healthy_when_model_loadable(tmp_path):
    module = _load_app_module(tmp_path)
    module.MODEL_DIR = str(tmp_path / "model")
    module._cached_model = None
    mlmodel_path = Path(module.MODEL_DIR) / "MLmodel"
    mlmodel_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel_path.write_text("artifact_path: model\n", encoding="utf-8")
    module.try_load_model = lambda: object()
    client = module.app.test_client()

    response = client.get("/ping")

    assert response.status_code == 200
