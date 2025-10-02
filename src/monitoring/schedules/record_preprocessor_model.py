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


def _is_number(v) -> bool:
    """Return True if v looks like a finite number."""
    return isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v))


# ---- add this small helper near the top of the file ----
def _attach_event_meta(rec: dict, inference_record) -> dict:
    """
    Attach event metadata/version to the output record so that the analyzer
    can join with ground-truth on eventId.
    """
    # Some containers expose 'event_metadata' and 'event_version' attributes.
    meta = getattr(inference_record, "event_metadata", None)
    ver = getattr(inference_record, "event_version", None)

    # Only attach if they are present and of the right types.
    if isinstance(meta, dict):
        # Must include eventId inside this dict for the join to work.
        rec["eventMetadata"] = meta
    if ver is not None:
        # Keep it as string for consistency.
        rec["eventVersion"] = str(ver)
    return rec


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

    # Parse endpoint_output first (works for output-only merged datasets)
    out_enc = getattr(inference_record.endpoint_output, "encoding", None)
    out_raw = getattr(inference_record.endpoint_output, "data", "") or ""
    out_raw = out_raw.rstrip("\n")
    pred_scalar = None
    pred_series = None
    out_payload = None
    if out_enc == "JSON" and out_raw:
        out_payload = _safe_json_loads(out_raw)
        if isinstance(out_payload, dict) and "predictions" in out_payload:
            preds = out_payload["predictions"]
            if isinstance(preds, list):
                # Prefer vector if all numeric
                if all(_is_number(x) for x in preds):
                    pred_series = [float(x) for x in preds]
            if pred_series is None:
                pred_scalar = _extract_prediction_scalar(out_payload)

    # Try to parse endpoint_input (optional; may be absent with endpointOutput-only merge)
    in_enc = getattr(inference_record.endpoint_input, "encoding", None)
    in_raw = getattr(inference_record.endpoint_input, "data", "") or ""
    in_raw = in_raw.rstrip("\n")
    cols, rows = [], []
    if in_enc == "JSON" and in_raw:
        in_payload = _safe_json_loads(in_raw)
        cols, rows = _extract_rows_from_dataframe_split(in_payload)

    # Build output records
    results = []

    # Case A: we have input rows (features). Attach predictions per-row if available.
    if rows:
        for idx, r in enumerate(rows):
            rec = {c: r[i] for i, c in enumerate(cols)}
            val = None
            if pred_series is not None and idx < len(pred_series):
                val = pred_series[idx]
            elif pred_scalar is not None:
                val = pred_scalar

            if _is_number(val):
                valf = float(val)
                # Emit both keys to accommodate analyzer expectations
                rec["predictions"] = valf
                rec["endpointOutput_predictions"] = valf
                rec = _attach_event_meta(rec, inference_record)
                results.append(rec)
        if not results:
            return []
        return results[0] if len(results) == 1 else results

    # Case B: no input rows (output-only). Emit records from output payload.
    if pred_series is not None and len(pred_series) > 0:
        for v in pred_series:
            if _is_number(v):
                valf = float(v)
                results.append(
                    _attach_event_meta(
                        {
                            "predictions": valf,
                            "endpointOutput_predictions": valf,
                        },
                        inference_record,
                    )  # <-- attach meta
                )
        return results  # may be multi-record

    if _is_number(pred_scalar):
        valf = float(pred_scalar)
        out = {"predictions": valf, "endpointOutput_predictions": valf}
        return _attach_event_meta(out, inference_record)

    # Nothing usable; skip
    return []
