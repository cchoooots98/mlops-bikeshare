# monitoring/mq_postprocessor.py
# Purpose: Normalize endpoint_output that looks like {"predictions": [p1, p2, ...]}
# into a list of dicts: [{"yhat": p1}, {"yhat": p2}, ...], so Model Quality can
# align per-row predictions against labels.

import json


def postprocess_handler(inference_record):
    """
    Called once per captured request/response pair.
    We only read endpoint_output here and return either:
      - dict {"yhat": float} for a single prediction, OR
      - list[dict] [{"yhat": float}, ...] for multiple predictions.
    """
    raw = (inference_record.endpoint_output.data or "").strip()

    # The capture is JSON: {"predictions":[...]}
    try:
        obj = json.loads(raw)
    except Exception:
        return {}  # not JSON; let the job skip this record

    preds = None
    if isinstance(obj, dict) and "predictions" in obj:
        preds = obj["predictions"]

    # If predictions are present as a list of numbers, expand them
    if isinstance(preds, list) and preds:
        out = []
        for v in preds:
            try:
                out.append({"yhat": float(v)})
            except Exception:
                # skip bad values
                continue
        return out if out else {}

    # If it is a single number or different shape, try best-effort fallback
    if isinstance(obj, (int, float)):
        return {"yhat": float(obj)}

    # Nothing usable
    return {}
