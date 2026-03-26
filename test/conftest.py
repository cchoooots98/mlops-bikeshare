import os
import sys
import types
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _install_mlflow_stub() -> None:
    if "mlflow" in sys.modules:
        return

    mlflow = types.ModuleType("mlflow")
    mlflow._tracking_uri = None
    mlflow._run_counter = 0
    mlflow._model_version_counter = 0
    mlflow._logged_dicts = {}

    def set_tracking_uri(uri):
        mlflow._tracking_uri = uri

    def set_experiment(_name):
        return None

    def autolog(disable=True):
        return None

    def set_tags(_tags):
        return None

    def log_params(_params):
        return None

    def log_metrics(_metrics):
        return None

    def log_text(_text, _artifact_file):
        return None

    def log_figure(_figure, _artifact_file):
        return None

    def log_dict(data, artifact_path):
        mlflow._logged_dicts[artifact_path] = data

    class _RunContext:
        def __enter__(self):
            mlflow._run_counter += 1
            self.info = types.SimpleNamespace(run_id=f"stub-run-{mlflow._run_counter}")
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def start_run(run_name=None):
        return _RunContext()

    def register_model(model_uri, name):
        mlflow._model_version_counter += 1
        return types.SimpleNamespace(version=str(mlflow._model_version_counter), model_uri=model_uri, name=name)

    class _PythonModel:
        pass

    def _pyfunc_log_model(artifact_path, python_model, signature=None, input_example=None):
        mlflow._logged_model = {
            "artifact_path": artifact_path,
            "python_model": python_model,
            "signature": signature,
            "input_example": input_example,
        }

    def _pyfunc_save_model(path, python_model, signature=None, input_example=None):
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "MLmodel").write_text("artifact_path: model\nflavors: {}\n", encoding="utf-8")
        mlflow._saved_model = {
            "path": str(target),
            "python_model": python_model,
            "signature": signature,
            "input_example": input_example,
        }

    class _MlflowClient:
        def transition_model_version_stage(self, name, version, stage, archive_existing_versions=False):
            return None

        def set_registered_model_alias(self, model_name, alias, version):
            return None

        def set_model_version_tag(self, model_name, version, key, value):
            return None

    def _download_artifacts(run_id=None, artifact_path=None):
        target = Path(REPO_ROOT) / artifact_path
        if target.exists():
            return str(target)
        raise FileNotFoundError(f"stub mlflow cannot find artifact_path={artifact_path} run_id={run_id}")

    mlflow.set_tracking_uri = set_tracking_uri
    mlflow.set_experiment = set_experiment
    mlflow.autolog = autolog
    mlflow.set_tags = set_tags
    mlflow.log_params = log_params
    mlflow.log_metrics = log_metrics
    mlflow.log_text = log_text
    mlflow.log_figure = log_figure
    mlflow.log_dict = log_dict
    mlflow.start_run = start_run
    mlflow.register_model = register_model
    mlflow.pyfunc = types.SimpleNamespace(
        PythonModel=_PythonModel, log_model=_pyfunc_log_model, save_model=_pyfunc_save_model
    )
    mlflow.tracking = types.SimpleNamespace(MlflowClient=_MlflowClient)
    mlflow.artifacts = types.SimpleNamespace(download_artifacts=_download_artifacts)

    signature_module = types.ModuleType("mlflow.models.signature")
    signature_module.infer_signature = lambda *args, **kwargs: {"args": args, "kwargs": kwargs}

    models_module = types.ModuleType("mlflow.models")
    models_module.signature = signature_module

    pyfunc_module = types.ModuleType("mlflow.pyfunc")
    pyfunc_module.PythonModel = _PythonModel
    pyfunc_module.log_model = _pyfunc_log_model
    pyfunc_module.save_model = _pyfunc_save_model

    tracking_module = types.ModuleType("mlflow.tracking")
    tracking_module.MlflowClient = _MlflowClient

    artifacts_module = types.ModuleType("mlflow.artifacts")
    artifacts_module.download_artifacts = _download_artifacts

    sys.modules["mlflow"] = mlflow
    sys.modules["mlflow.pyfunc"] = pyfunc_module
    sys.modules["mlflow.tracking"] = tracking_module
    sys.modules["mlflow.artifacts"] = artifacts_module
    sys.modules["mlflow.models"] = models_module
    sys.modules["mlflow.models.signature"] = signature_module


try:
    import mlflow  # noqa: F401
except ModuleNotFoundError:
    _install_mlflow_stub()
