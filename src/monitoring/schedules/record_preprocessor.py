# record_preprocessor.py
# This script is invoked per record by the Model Monitor container.
# It must define a function named `preprocess_handler(inference_record)`.

import json


def preprocess_handler(inference_record):
    """
    Parameters
    ----------
    inference_record: An object with endpoint_input/endpoint_output/metadata.
                      We only rely on endpoint_input here.
    Returns
    -------
    dict or list[dict]:
        A flattened mapping of features for ONE record (or a list of such dicts).
        Model Monitor will treat keys as feature names and values as scalars.
    """
    enc = inference_record.endpoint_input.encoding
    raw = inference_record.endpoint_input.data.rstrip("\n")

    # Only handle JSON captured requests here; skip others gracefully.
    if enc != "JSON":
        return []

    try:
        payload = json.loads(raw)
        # Expect the standard SM capture format with dataframe_split
        cols = payload["inputs"]["dataframe_split"]["columns"]
        rows = payload["inputs"]["dataframe_split"]["data"]
    except Exception:
        # If format is unexpected, skip this record
        return []

    # Convert the first row to {feature_name: value} (or return a list if multiple rows)
    if not rows:
        return []

    # If multiple rows are present in one record, return a list of dicts.
    result = [{c: r[i] for i, c in enumerate(cols)} for r in rows]
    return result if len(result) > 1 else result[0]
