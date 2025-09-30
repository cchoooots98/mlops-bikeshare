# src/inference/lambda_handler.py
# All comments in English.

import os
import json

def handler(event, context):
    """
    Lambda entrypoint.
    We import heavy modules INSIDE the handler so that:
      - We can log versions (e.g., sklearn) for debugging.
      - We avoid import-time failures at module load stage.
    """
    # Optional: allow overriding endpoint name from event
    if isinstance(event, dict) and "sm_endpoint" in event:
        os.environ["SM_ENDPOINT"] = str(event["sm_endpoint"])

    # --- Debug: verify sklearn is importable in Lambda runtime ---
    try:
        import sklearn  # lazy import to confirm presence in the image
        skl_ver = getattr(sklearn, "__version__", "unknown")
    except Exception as e:
        # Return a helpful error so CloudWatch shows the real cause
        return {"ok": False, "error": f"sklearn import failed: {repr(e)}"}

    # Import predictor lazily (it imports build_features -> sklearn BallTree)
    from src.inference import predictor

    # Run one full prediction cycle
    predictor.main()

    return {
        "ok": True,
        "endpoint": os.environ.get("SM_ENDPOINT", "bikeshare-staging"),
        "sklearn": skl_ver,
        "message": "predictor finished one cycle"
    }
