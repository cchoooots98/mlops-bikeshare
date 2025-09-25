# record_preprocessor_model.py
# -----------------------------------------------------------
# Purpose:
#   Transform each captured inference record into a flat dict (or list of dicts)
#   that includes BOTH features (from endpoint_input) and a scalar prediction
#   under the key "predictions" (from endpoint_output), so that
#   ProbabilityAttribute="predictions" works as expected.
#
# Notes:
#   - This function is invoked by the Model Monitor container.
#   - It MUST define `preprocess_handler(inference_record)`.
#   - It should return either a single dict or a list[dict].
#
# Works with:
#   - Request: JSON, SM "dataframe_split" format under payload["inputs"]["dataframe_split"]
#   - Response: JSON with either {"predictions": <number>} or {"predictions": [<number>, ...]}
#
# Windows/VS Code friendly; no 3rd-party deps.
# -----------------------------------------------------------

import json


def _safe_json_loads(text: str):
    """Safely parse JSON; return None on failure."""
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_rows_from_dataframe_split(payload: dict):
    """
    Extract (columns, rows) from SageMaker dataframe_split request structure.
    Returns: (columns: list[str], rows: list[list]) or ([], [])
    """
    try:
        ds = payload["inputs"]["dataframe_split"]
        cols = ds["columns"]
        rows = ds["data"]
        if isinstance(cols, list) and isinstance(rows, list):
            return cols, rows
    except Exception:
        pass
    return [], []


def _extract_prediction_scalar(resp_payload: dict):
    """
    Extract a scalar prediction from endpoint_output payload.
    Accepts:
      {"predictions": number}                      -> returns number
      {"predictions": [number, ...]}              -> returns number (first element)
      {"predictions": [[number], ...]} (rare)     -> returns number (first leaf)
    Returns None if not found/parseable.
    """
    if not isinstance(resp_payload, dict):
        return None

    if "predictions" not in resp_payload:
        return None

    pred = resp_payload["predictions"]

    # Case 1: already a scalar number
    if isinstance(pred, (int, float)):
        return float(pred)

    # Case 2: list/array-like -> take first numeric leaf
    if isinstance(pred, list) and pred:
        first = pred[0]
        if isinstance(first, (int, float)):
            return float(first)
        if isinstance(first, list) and first and isinstance(first[0], (int, float)):
            return float(first[0])

    # Not a supported shape
    return None


def preprocess_handler(inference_record):
    """
    Parameters
    ----------
    inference_record: An object with endpoint_input / endpoint_output / event_metadata.
                      See AWS docs for available fields.

    Returns
    -------
    dict or list[dict]:
        A flat mapping or a list of flat mappings. Each mapping MUST include
        a scalar "predictions" key so that ProbabilityAttribute="predictions" works.
    """
    # 1) Only handle JSON input; skip otherwise gracefully.
    in_enc = getattr(inference_record.endpoint_input, "encoding", None)
    in_raw = getattr(inference_record.endpoint_input, "data", "") or ""
    in_raw = in_raw.rstrip("\n")

    if in_enc != "JSON" or not in_raw:
        # No usable input; return empty to skip this record.
        return []

    # 2) Parse request payload and extract dataframe_split
    in_payload = _safe_json_loads(in_raw)
    cols, rows = _extract_rows_from_dataframe_split(in_payload)
    if not cols or not rows:
        # Unexpected request shape; skip
        return []

    # 3) Parse response payload and extract a scalar probability
    out_enc = getattr(inference_record.endpoint_output, "encoding", None)
    out_raw = getattr(inference_record.endpoint_output, "data", "") or ""
    out_raw = out_raw.rstrip("\n")
    pred_scalar = None
    pred_series = None  # optional per-row predictions

    if out_enc == "JSON" and out_raw:
        out_payload = _safe_json_loads(out_raw)

        # If predictions is a vector and length matches number of rows,
        # we can keep it as a per-row list; otherwise fall back to scalar.
        if (
            isinstance(out_payload, dict)
            and "predictions" in out_payload
            and isinstance(out_payload["predictions"], list)
        ):
            if len(out_payload["predictions"]) == len(rows) and all(
                isinstance(x, (int, float)) for x in out_payload["predictions"]
            ):
                pred_series = [float(x) for x in out_payload["predictions"]]

        # Always try to produce a scalar fallback
        if pred_series is None:
            pred_scalar = _extract_prediction_scalar(out_payload)

    # 4) Build output record(s)
    results = []
    for idx, r in enumerate(rows):
        # Map features
        rec = {c: r[i] for i, c in enumerate(cols)}

        # Attach prediction as scalar
        if pred_series is not None:
            rec["predictions"] = pred_series[idx] if idx < len(pred_series) else None
        else:
            rec["predictions"] = pred_scalar

        results.append(rec)

    # If only one row, return dict; otherwise, a list of dicts
    return results[0] if len(results) == 1 else results
