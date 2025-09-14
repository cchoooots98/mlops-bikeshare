# handler.py
# Calls SageMaker endpoint with the latest online features and writes:
# 1) s3://.../inference/city=.../dt=.../predictions.parquet
# 2) After 30 minutes, builds actuals from v_station_status, joins with predictions,
#    and writes s3://.../monitoring/quality/city=.../ds=YYYY-MM-DD/part-*.parquet
# Run it every 5–10 minutes via GitHub Actions (cron) or locally.

import io
import json
import os
from datetime import datetime, timedelta, timezone

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.features.build_features import athena_conn, query_df, read_env  # reuse env + athena

# Import the online featurizer and shared schema
# REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if REPO_ROOT not in sys.path:
#     sys.path.insert(0, REPO_ROOT)
from src.features.schema import FEATURE_COLUMNS  # same order as training
from src.inference.featurize_online import build_online_features  # latest feature batch


def _s3():
    return boto3.client("s3")


def _smr():
    # SageMaker runtime client for InvokeEndpoint
    return boto3.client("sagemaker-runtime")


def _write_parquet_s3(df: pd.DataFrame, bucket: str, key: str):
    # Write a small DataFrame to S3 as parquet
    # (in-memory buffer to avoid temp files on Windows)
    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    _s3().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


def _inference_table_create_if_absent(cnx, bucket):
    # External table for predictions (partitioned by city, dt)
    sql = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS inference (
        station_id string,
        yhat_bikes double,            
        raw string                 
    )
    PARTITIONED BY (`city` string, `dt` string)
    STORED AS PARQUET
    LOCATION 's3://{bucket}/inference/'
    TBLPROPERTIES ('parquet.compression'='SNAPPY')
    """
    pd.read_sql(sql, cnx)
    pd.read_sql("MSCK REPAIR TABLE inference", cnx)


def _quality_table_create_if_absent(cnx, bucket):
    # External table for monitoring join
    sql = """
    CREATE EXTERNAL TABLE IF NOT EXISTS monitoring_quality (
        station_id string,
        dt string,            
        dt_plus30 string,     
        yhat_bikes double,
        y_stockout_bikes_30 double,
        bikes_t30 int
        )
    PARTITIONED BY (city string, ds string)    
    STORED AS PARQUET
    LOCATION 's3://mlops-bikeshare-387706002632-ca-central-1/monitoring/quality/'
    TBLPROPERTIES ('parquet.compression'='SNAPPY');
    """
    pd.read_sql(sql, cnx)
    pd.read_sql("MSCK REPAIR TABLE monitoring_quality", cnx)


def _invoke_endpoint(endpoint_name: str, X: pd.DataFrame) -> pd.DataFrame:
    """
    Invoke an MLflow pyfunc SageMaker endpoint.
    We send a pandas 'dataframe_split' payload to preserve column order and types.
    If your Step 6 container expects a different schema, adjust here.
    """
    payload = {
        "inputs": {
            "dataframe_split": {"columns": FEATURE_COLUMNS, "data": X[FEATURE_COLUMNS].astype(float).values.tolist()}
        }
    }
    resp = _smr().invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    body = resp["Body"].read()
    try:
        out = json.loads(body.decode("utf-8"))
    except Exception:
        raise RuntimeError(f"Bad model response: {body[:500]}")

    # Flexible parsing: support either a scalar list or dict/list of dicts
    # Normalize to a single 'yhat' column; keep 'raw'  for debugging.
    if isinstance(out, dict) and "predictions" in out:
        preds = out["predictions"]
    else:
        preds = out

    print(f"[DEBUG_hander] the out: {out}")

    # Heuristics:
    def to_scalar(x):
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, list) and len(x) == 1 and isinstance(x[0], (int, float)):
            return float(x[0])
        if isinstance(x, dict) and "yhat" in x:
            return float(x["yhat"])
        # fallback: NaN
        return float("nan")

    yhat = [to_scalar(p) for p in preds]
    res = X[["city", "dt", "station_id"]].copy()
    res["yhat_bikes"] = yhat
    res["raw"] = pd.Series([json.dumps(p, ensure_ascii=False) for p in preds], dtype="string")

    return res


def _compute_actuals_for_dt(cnx, city: str, pred_dt: str, threshold: int = 2) -> pd.DataFrame:
    """
    Build t+30m actuals from v_station_status, then compute label:
    y_stockout_bikes_30 = 1.0 if bikes(t+30m) <= threshold else 0.0
    """
    # Compute t+30 string and fetch bikes/docks at t+30 for all stations
    dt_plus30 = (datetime.strptime(pred_dt, "%Y-%m-%d-%H-%M") + timedelta(minutes=30)).strftime("%Y-%m-%d-%H-%M")
    sql = f"""
    SELECT station_id, bikes AS bikes_t30
    FROM {cnx.schema_name}.v_station_status
    WHERE city = '{city}' AND dt = '{dt_plus30}'
    """
    df = query_df(cnx, sql)

    if df.empty:
        # Not ready yet; a future run will fill this in.
        return pd.DataFrame(columns=["station_id", "bikes_t30", "y_stockout_bikes_30", "dt_plus30"])

    df["y_stockout_bikes_30"] = (df["bikes_t30"] <= threshold).astype("float32")
    df["dt_plus30"] = dt_plus30
    return df


def main():
    # Read config
    cfg = read_env()
    city = cfg["city"]
    bucket = cfg["bucket"]

    # You can switch between staging/prod via env or CLI args (simplest: set here)
    endpoint_name = os.environ.get("SM_ENDPOINT", "bikeshare-prod")

    # Prepare Athena connection
    cnx = athena_conn(
        region=cfg["region"],
        s3_staging_dir=cfg["athena_output"],
        workgroup=cfg["athena_workgroup"],
        schema_name=cfg["athena_database"],
    )

    # Ensure external tables exist (idempotent)
    _inference_table_create_if_absent(cnx, bucket)
    _quality_table_create_if_absent(cnx, bucket)

    # === A. Produce predictions for the latest snapshot ===
    X = build_online_features(city)  # includes ["city","dt","station_id"] + FEATURE_COLUMNS
    latest_dt = X["dt"].iloc[0]

    preds = _invoke_endpoint(endpoint_name, X)

    # Write to S3 partition: inference/city=.../dt=.../predictions.parquet
    pred_key = f"inference/city={city}/dt={latest_dt}/predictions.parquet"
    _write_parquet_s3(preds[["station_id", "yhat_bikes", "raw"]], bucket, pred_key)

    # Repair partitions (lightweight)
    try:
        pd.read_sql("MSCK REPAIR TABLE inference", cnx)
    except Exception:
        pass

    # === B. Backfill actuals for any predictions older than 30 minutes ===
    # Strategy: find inference partitions (last ~2 hours), join those with missing quality rows.
    # For simplicity, compute for *this* latest_dt if it is older than now-30m,
    # and also try the previous 12 intervals to reduce gaps.
    now_utc = datetime.now(timezone.utc)
    candidate_dts = []
    for k in range(0, 13):  # ~last 60 minutes
        dtk = (datetime.strptime(latest_dt, "%Y-%m-%d-%H-%M") - timedelta(minutes=5 * k)).strftime("%Y-%m-%d-%H-%M")
        # Only attempt if t+30m should already exist
        if now_utc >= (datetime.strptime(dtk, "%Y-%m-%d-%H-%M").replace(tzinfo=timezone.utc) + timedelta(minutes=30)):
            candidate_dts.append(dtk)

    # Load any predictions for candidate dts and fill quality if actuals exist
    for dt_pred in candidate_dts:
        # Read preds back via S3 Select would add complexity; we already have them in memory for latest,
        # but for older we just try to compute actuals then join with what we just wrote for 'latest_dt'.
        # Practical simplification: only finalize 'latest_dt' quality here; a nightly job can fully backfill.
        if dt_pred != latest_dt:
            continue

        actuals = _compute_actuals_for_dt(cnx, city, dt_pred, threshold=2)
        if actuals.empty:
            continue

        # Join preds+actuals on station_id
        joined = preds.merge(
            actuals[["station_id", "bikes_t30", "y_stockout_bikes_30", "dt_plus30"]], on="station_id", how="inner"
        ).assign(dt=lambda d: dt_pred)

        # Write to monitoring/quality partitioned by city, ds (ds = date of pred time)
        ds = dt_pred[:10]  # YYYY-MM-DD
        qual_key = f"monitoring/quality/city={city}/ds={ds}/part-{dt_pred}.parquet"
        _write_parquet_s3(
            joined[["station_id", "dt", "dt_plus30", "yhat_bikes", "y_stockout_bikes_30", "bikes_t30"]],
            bucket,
            qual_key,
        )
        try:
            pd.read_sql("MSCK REPAIR TABLE monitoring_quality", cnx)
        except Exception:
            pass


if __name__ == "__main__":
    # PowerShell example:
    # PS> $env:SM_ENDPOINT="bikeshare-prod"; python src/inference/handler.py
    main()
