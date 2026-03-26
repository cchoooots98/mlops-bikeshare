from __future__ import annotations

import os

from src.config import endpoint_name
from src.serving.request_contract import parse_router_request
from src.serving.resolver import resolve_deployment_state_path


def _apply_endpoint_override(event) -> None:
    request = parse_router_request(event)
    os.environ["TARGET_NAME"] = request.target_name
    os.environ["PREDICT_BIKES"] = str(request.predict_bikes).lower()
    os.environ["SERVING_ENVIRONMENT"] = request.environment
    os.environ["DEPLOYMENT_STATE_PATH"] = str(
        resolve_deployment_state_path(
            target_name=request.target_name,
            environment=request.environment,
            deployment_state_root=os.getenv("DEPLOYMENT_STATE_ROOT", "model_dir/deployments"),
        )
    )
    if request.sm_endpoint_override:
        os.environ["SM_ENDPOINT"] = request.sm_endpoint_override
        return
    os.environ["SM_ENDPOINT"] = endpoint_name(target_name=request.target_name, environment=request.environment)


def _validate_dependencies() -> dict:
    try:
        import pandas
        import sklearn

        return {
            "ok": True,
            "mode": "validate_only",
            "sklearn": getattr(sklearn, "__version__", "unknown"),
            "pandas": getattr(pandas, "__version__", "unknown"),
        }
    except Exception as exc:
        return {"ok": False, "mode": "validate_only", "error": repr(exc)}


def handler(event, context):
    """Serving entrypoint for the predictor Lambda."""
    request = parse_router_request(event)
    _apply_endpoint_override(event)
    if request.validate_only:
        return _validate_dependencies()

    try:

        from src.inference import predictor

        predictor.main()
        return {
            "ok": True,
            "endpoint": os.environ.get(
                "SM_ENDPOINT", endpoint_name(target_name=request.target_name, environment=request.environment)
            ),
            "prediction_target": request.target_name,
            "message": "predictor finished one cycle",
        }
    except Exception as exc:
        return {"ok": False, "prediction_target": request.target_name, "error": repr(exc)}
