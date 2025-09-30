# src/inference/lambda_handler.py
# All comments in English.
import os


def handler(event, context):
    """Lambda entrypoint with a fast 'validate_only' path."""
    # 1) Optional: override endpoint from event
    if isinstance(event, dict) and "sm_endpoint" in event:
        os.environ["SM_ENDPOINT"] = str(event["sm_endpoint"])

    # 2) Fast path: validate imports only
    if isinstance(event, dict) and event.get("validate_only"):
        try:
            import pandas
            import sklearn

            return {
                "ok": True,
                "mode": "validate_only",
                "sklearn": getattr(sklearn, "__version__", "unknown"),
                "pandas": getattr(pandas, "__version__", "unknown"),
            }
        except Exception as e:
            return {"ok": False, "mode": "validate_only", "error": repr(e)}

    # 3) Normal long run
    try:
        import sklearn  # lazy import to ensure present

        from src.inference import predictor  # imports build_features -> sklearn BallTree

        predictor.main()
        return {
            "ok": True,
            "endpoint": os.environ.get("SM_ENDPOINT", "bikeshare-staging"),
            "message": "predictor finished one cycle",
        }
    except Exception as e:
        return {"ok": False, "error": repr(e)}
