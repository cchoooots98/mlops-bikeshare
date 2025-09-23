# handler.py
# Calls SageMaker endpoint with the latest online features and writes:
# 1) s3://.../inference/city=.../dt=.../predictions.parquet
# 2) After 30 minutes, builds actuals from v_station_status, joins with predictions,
#    and writes s3://.../monitoring/quality/city=.../ds=YYYY-MM-DD/part-*.parquet
# Run it every 5–10 minutes via GitHub Actions (cron) or locally.

import io
import json
import os
import time
import warnings
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import gzip
from typing import Dict, Iterable, Tuple

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

from src.features.build_features import athena_conn, query_df, read_env  # reuse env + athena
from src.features.schema import FEATURE_COLUMNS  # same order as training
from src.inference.featurize_online import build_online_features  # latest feature batch

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")


TZ_LOCAL = ZoneInfo("America/New_York")
TZ_UTC = ZoneInfo("UTC")

YHAT_PROB_THRESHOLD = 0.15


def _s3():
    return boto3.client("s3")


def _smr():
    # SageMaker runtime client for InvokeEndpoint
    return boto3.client("sagemaker-runtime")

def _discover_capture_prefix(bucket: str, endpoint_name: str) -> str:
    """
    自动发现 Data Capture 的根前缀，目标形如：
    datacapture/endpoint=<endpoint-config>/<endpoint_name>/AllTraffic

    策略：
      1) 列出 datacapture/ 下的一级目录，筛选以 'endpoint=' 开头的目录；
      2) 在每个 endpoint=<...>/ 下面查找是否存在 '<endpoint_name>/AllTraffic/'；
      3) 选择最近修改的一个作为前缀（更稳妥地对应最新 endpoint-config）。
    """
    s3 = _s3()
    root = "datacapture/"
    # 先枚举 endpoint=* 目录
    token = None
    cand_prefixes = []
    while True:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=root, Delimiter="/",
                                  ContinuationToken=token) if token else \
               s3.list_objects_v2(Bucket=bucket, Prefix=root, Delimiter="/")
        for cp in resp.get("CommonPrefixes", []):
            pfx = cp.get("Prefix")  # e.g. 'datacapture/endpoint=xxx/'
            if pfx and pfx.startswith("datacapture/endpoint="):
                cand_prefixes.append(pfx)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break

    best = None
    best_mtime = None

    # 在每个 endpoint=* 下面找 '<endpoint_name>/AllTraffic/'
    for ep_root in cand_prefixes:
        target = f"{ep_root}{endpoint_name}/AllTraffic/"
        # 看这个 target 下是否有文件
        token = None
        found_any = False
        last_mtime = None
        while True:
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=target, MaxKeys=1,
                                      ContinuationToken=token) if token else \
                   s3.list_objects_v2(Bucket=bucket, Prefix=target, MaxKeys=1)
            contents = resp.get("Contents", [])
            if contents:
                found_any = True
                # 记录这个 capture 路径下最新文件时间，作为“最近”的依据
                last_mtime = contents[0].get("LastModified")
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
            else:
                break

        if found_any:
            if best is None or (last_mtime and (best_mtime is None or last_mtime > best_mtime)):
                best = target.rstrip("/")  # 去掉尾部斜杠，和我们其余逻辑对齐
                best_mtime = last_mtime

    if not best:
        # 兜底：返回最常见的形态，仍然按你给出的目录结构
        # 这里不包含 endpoint-config，因为我们没找到具体的；调用方可选择报 warn
        return f"datacapture/endpoint=<UNKNOWN>/{endpoint_name}/AllTraffic"
    return best



def _read_parquet_s3(bucket: str, key: str) -> pd.DataFrame:
    """
    Read a small Parquet file from S3 into a pandas DataFrame.
    This is used to backfill older dt predictions for joining with actuals.
    """
    obj = _s3().get_object(Bucket=bucket, Key=key)
    buf = io.BytesIO(obj["Body"].read())
    table = pq.read_table(buf)
    return table.to_pandas()


def _write_parquet_s3(df: pd.DataFrame, bucket: str, key: str):
    # Write a small DataFrame to S3 as parquet
    # (in-memory buffer to avoid temp files on Windows)
    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    _s3().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())

def _list_s3_keys(bucket: str, prefix: str) -> Iterable[str]:
    """List all S3 object keys under prefix (non-recursive)."""
    s3 = _s3()
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for it in resp.get("Contents", []):
            yield it["Key"]
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break


def _read_s3_text(bucket: str, key: str) -> Iterable[str]:
    """Stream lines from a (possibly gzip) text object in S3."""
    obj = _s3().get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    # Many capture files are .jsonl or .jsonl.gz；两种都兼容
    if key.endswith(".gz"):
        try:
            body = gzip.decompress(body)
        except OSError:
            pass
    for line in body.splitlines():
        yield line.decode("utf-8", errors="ignore")


def _capture_hours_around(dt_str: str, window_minutes: int = 10) -> Iterable[str]:
    """
    给定一次预测的 dt(UTC 字符串 'YYYY-MM-DD-HH-MM')，返回应扫描的 capture 小时文件夹列表（字符串 'YYYY/MM/DD/HH'）。
    取 dt 所在小时，外加前/后各 1 小时，避免边界卡点。
    """
    base = datetime.strptime(dt_str, "%Y-%m-%d-%H-%M").replace(tzinfo=timezone.utc)
    hours = {
        base.strftime("%Y/%m/%d/%H"),
        (base - timedelta(hours=1)).strftime("%Y/%m/%d/%H"),
        (base + timedelta(hours=1)).strftime("%Y/%m/%d/%H"),
    }
    return sorted(hours)


def _build_inferenceid_to_eventid_map(
    bucket: str,
    capture_prefix: str,
    hour_keys: Iterable[str],
    limit_files_per_hour: int = 200,
) -> Dict[str, str]:
    """
    读取指定小时段下的 capture jsonl，建立 {inferenceId -> eventId} 映射。
    只使用包含两者字段的行，遇到异常/脏行跳过。
    """
    m: Dict[str, str] = {}
    for hour_key in hour_keys:
        prefix = f"{capture_prefix}/{hour_key}/"
        seen = 0
        for key in _list_s3_keys(bucket, prefix):
            # 限流，防止小时下文件极多
            if limit_files_per_hour and seen >= limit_files_per_hour:
                break
            if not (key.endswith(".jsonl") or key.endswith(".jsonl.gz") or key.endswith(".json")):
                continue
            seen += 1
            for line in _read_s3_text(bucket, key):
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                meta = rec.get("eventMetadata") or {}
                inf_id = meta.get("inferenceId")
                ev_id = meta.get("eventId")
                # 只要二者同时存在才可用于映射
                if isinstance(inf_id, str) and isinstance(ev_id, str):
                    # 第一次出现的映射采用；后续重复的同 key 忽略
                    m.setdefault(inf_id, ev_id)
    return m


def write_ground_truth_jsonl(s3_client, bucket: str, gt_root_prefix: str, rows):
    """
    Write Ground Truth JSONL for SageMaker Model Monitor (ModelQuality).
    - bucket: your S3 bucket name (string).
    - gt_root_prefix: S3 prefix whose immediate children are YYYY directories,
      e.g. "monitoring/ground-truth" (DO NOT use 'latest/').
    - rows: iterable of tuples (inference_id: str, label: int|bool).
    Output path: s3://bucket/gt_root_prefix/YYYY/MM/DD/HH/labels-<timestamp>.jsonl
    The 'YYYY/MM/DD/HH' MUST be the UTC hour of *label collection time*
    (see official doc).
    """
    # Use UTC "now" as the label collection time per AWS docs
    now_utc = datetime.now(timezone.utc)
    key = f"{gt_root_prefix}/{now_utc:%Y/%m/%d/%H}/labels-{now_utc:%Y%m%d%H%M%S}.jsonl"

    # Build lines using the EXACT schema required by AWS:
    # {
    #  "groundTruthData":{"data":"0|1","encoding":"CSV"},
    #  "eventMetadata":{"eventId":"<inference_id>"},
    #  "eventVersion":"0"
    # }
    buf = io.StringIO()
    for inference_id, label in rows:
        rec = {
            "groundTruthData": {"data": str(int(label)), "encoding": "CSV"},
            "eventMetadata": {"eventId": str(inference_id)},
            "eventVersion": "0",
        }
        buf.write(json.dumps(rec) + "\n")

    s3_client.put_object(Bucket=bucket, Key=key, Body=buf.getvalue().encode("utf-8"), ContentType="application/json")
    return f"s3://{bucket}/{key}"


def _inference_table_create_if_absent(cnx, bucket):
    # External table for predictions (partitioned by city, dt)
    sql = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS inference (
        station_id string,
        yhat_bikes double,
        yhat_bikes_bin double,
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
        yhat_bikes_bin double,
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


def _invoke_with_retry(rt, **kwargs):
    """
    InvokeEndpoint with simple exponential backoff.
    Retries on ModelError / 5xx up to max_retries times.
    """
    max_retries = int(os.environ.get("SM_MAX_RETRIES", "5"))
    base = 0.5  # seconds
    for i in range(max_retries + 1):
        try:
            return rt.invoke_endpoint(**kwargs)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            # Retry on throttling / 5xx / model errors
            if i < max_retries and (
                code in ("ModelError", "ValidationError", "InternalFailure", "ThrottlingException")
            ):
                time.sleep(base * (2**i))  # 0.5,1,2,4,8...
                continue
            raise


def _invoke_endpoint_rowwise(endpoint_name: str, X: pd.DataFrame) -> pd.DataFrame:
    """
    Invoke the SageMaker endpoint one row at a time and attach an InferenceId per record.
    This enables Model Monitor (ModelQuality) to merge predictions with Ground Truth by 'inferenceId'.
    NOTE: This is slower than batching but is robust for joining.
    """
    rows = []
    rt = _smr()  # boto3.client("sagemaker-runtime")

    for rec in X[["city", "dt", "station_id"] + FEATURE_COLUMNS].itertuples(index=False, name=None):
        city, dt_str, station_id, *features = rec

        # Build dataframe_split payload with a single record to preserve column order
        payload = {
            "inputs": {
                "dataframe_split": {
                    "columns": FEATURE_COLUMNS,
                    "data": [list(map(float, features))],
                }
            }
        }

        # Deterministic inference id so the Ground Truth builder can regenerate it:
        inference_id = f"{dt_str}_{station_id}"

        resp = _invoke_with_retry(
            rt,
            EndpointName=endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            InferenceId=inference_id,
            Body=json.dumps(payload).encode("utf-8"),
        )

        body = resp["Body"].read()
        try:
            out = json.loads(body.decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"Bad model response for {inference_id}: {body[:500]}") from e

        # Normalize to a scalar probability from various common shapes
        def to_scalar(x):
            if isinstance(x, (int, float)):
                return float(x)
            if isinstance(x, list) and len(x) == 1 and isinstance(x[0], (int, float)):
                return float(x[0])
            if isinstance(x, dict) and "yhat" in x:
                return float(x["yhat"])
            return float("nan")

        preds = out["predictions"] if isinstance(out, dict) and "predictions" in out else out
        yhat = to_scalar(preds[0] if isinstance(preds, list) else preds)

        rows.append(
            {
                "city": city,
                "dt": dt_str,
                "station_id": station_id,
                "yhat_bikes": yhat,
                "yhat_bikes_bin": float(yhat >= YHAT_PROB_THRESHOLD),
                "inference_id": inference_id,
                "raw": json.dumps(out, ensure_ascii=False),
            }
        )
        sleep_ms = int(os.environ.get("ROWWISE_SLEEP_MS", "20"))  # 20ms default
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000)

    return pd.DataFrame(rows)

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

    df["y_stockout_bikes_30"] = (df["bikes_t30"] <= threshold).astype("float64")
    df["dt_plus30"] = dt_plus30
    return df


def main():
    # Read config
    cfg = read_env()
    city = cfg["city"]
    bucket = cfg["bucket"]

    # You can switch between staging/prod via env or CLI args (simplest: set here)
    endpoint_name = os.environ.get("SM_ENDPOINT", "bikeshare-staging")
    capture_prefix = os.environ.get("SM_CAPTURE_PREFIX")
    if not capture_prefix:
        # 自动发现形如 datacapture/endpoint=<...>/<endpoint_name>/AllTraffic
        capture_prefix = _discover_capture_prefix(bucket=bucket, endpoint_name=endpoint_name)
        print(f"[info] SM_CAPTURE_PREFIX not set. Auto-discovered capture_prefix={capture_prefix}")
    else:
        print(f"[info] Using SM_CAPTURE_PREFIX={capture_prefix}")
        
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
    max_rows = int(os.environ.get("MAX_ROWS_PER_RUN", "300"))  # process at most 300 rows per run
    if len(X) > max_rows:
        X = X.head(max_rows).copy()

    latest_dt = X["dt"].iloc[0]

    preds = _invoke_endpoint_rowwise(endpoint_name, X)

    # Write to S3 partition: inference/city=.../dt=.../predictions.parquet
    pred_key = f"inference/city={city}/dt={latest_dt}/predictions.parquet"
    _write_parquet_s3(preds[["station_id", "yhat_bikes", "yhat_bikes_bin", "inference_id", "raw"]], bucket, pred_key)

    # Repair partitions (lightweight)
    try:
        pd.read_sql("MSCK REPAIR TABLE inference", cnx)
    except Exception:
        pass

    # === Build candidate dt list (last ~60 minutes in 5-min steps) ===
    # Keep only those whose t+30 ground truth should already exist
    candidate_dts = []
    now_utc = datetime.now(timezone.utc)
    for k in range(0, 13):  # 0,5,10,...,60 minutes back
        dtk = (datetime.strptime(latest_dt, "%Y-%m-%d-%H-%M") - timedelta(minutes=5 * k)).strftime("%Y-%m-%d-%H-%M")
        if now_utc >= (datetime.strptime(dtk, "%Y-%m-%d-%H-%M").replace(tzinfo=timezone.utc) + timedelta(minutes=30)):
            candidate_dts.append(dtk)

    candidate_dts = sorted(set(candidate_dts))  # ensure unique + ascending (oldest -> newest)

    def _gt_hour_exists(s3_client, bucket: str, gt_root_prefix: str, y: str, m: str, d: str, h: str) -> bool:
        """
        Return True if there is at least one labels-*.jsonl under s3://bucket/gt_root_prefix/YYYY/MM/DD/HH/.
        This makes ground-truth writing idempotent and avoids duplicate hours.
        """
        prefix = f"{gt_root_prefix}/{y}/{m}/{d}/{h}/"
        resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        return "Contents" in resp

    # This dict aggregates ground-truth rows by the *dt_plus30 hour*.
    # Key   : "YYYY/MM/DD/HH" (UTC hour of the label, derived from dt_plus30)
    # Value : list of tuples [(inference_id, label_int), ...]
    hour_to_rows: Dict[str, list[Tuple[str, int]]] = {}

    # We will also keep track whether we wrote any parquet or labels this run
    wrote_any_parquet = False

    # === Iterate candidate dts, join predictions & actuals, write parquet for auditing, collect GT rows ===
    for dt_pred in candidate_dts:
        # Load predictions for this dt:
        # - reuse in-memory 'preds' if it's the current latest_dt
        # - otherwise read back from S3: inference/city=.../dt=.../predictions.parquet
        if dt_pred == latest_dt:
            preds_dt = preds.copy()
        else:
            pred_key_prev = f"inference/city={city}/dt={dt_pred}/predictions.parquet"
            try:
                preds_dt = _read_parquet_s3(bucket, pred_key_prev)
            except Exception as e:
                # Missing predictions parquet is not fatal for ModelQuality
                print(f"[warn] missing predictions parquet: s3://{bucket}/{pred_key_prev} ({e})")
                continue

        # Compute actuals for this dt; if not ready yet, skip and let a future run handle it
        actuals = _compute_actuals_for_dt(cnx, city, dt_pred, threshold=2)
        if actuals.empty:
            print(f"[info] actuals not ready for dt={dt_pred}; skip")
            continue

        # Join on station_id; keep inference_id (used by Model Monitor merge)
        joined = preds_dt.merge(
            actuals[["station_id", "bikes_t30", "y_stockout_bikes_30", "dt_plus30"]],
            on="station_id",
            how="inner",
        ).assign(dt=lambda d: dt_pred)

        # Write partitioned parquet for auditing / Athena
        ds = dt_pred[:10]  # YYYY-MM-DD
        qual_key = f"monitoring/quality/city={city}/ds={ds}/part-{dt_pred}.parquet"
        _write_parquet_s3(
            joined[
                [
                    "station_id",
                    "dt",
                    "dt_plus30",
                    "yhat_bikes",
                    "yhat_bikes_bin",
                    "y_stockout_bikes_30",
                    "bikes_t30",
                    "inference_id",
                ]
            ],
            bucket,
            qual_key,
        )
        wrote_any_parquet = True

        # === Aggregate Ground Truth rows by the hour of dt_plus30 (UTC) ===
        # Model Monitor allows the hour folders to represent "label collection time".
        # Here we use the dt_plus30 hour (when the truth is observed) as the hour bucket.
        # Example: if dt_pred=18:45, dt_plus30=19:15 -> hour bucket is 19.
        # This produces stable hour folders even when we backfill.
        if "dt_plus30" not in joined.columns:
            # Safety guard (should not happen if _compute_actuals_for_dt returns dt_plus30)
            continue

        # Parse the dt_plus30 string "YYYY-MM-DD-HH-mm" and get UTC hour components
        def _ymdh_from_utc_dt(dt_str: str):
            # Slice strings (no tz math needed because dt strings are already UTC)
            return dt_str[0:4], dt_str[5:7], dt_str[8:10], dt_str[11:13]

        capture_hours = _capture_hours_around(dt_pred)
        infid_to_evid = _build_inferenceid_to_eventid_map(bucket=bucket,
                                                          capture_prefix=capture_prefix,
                                                          hour_keys=capture_hours)
        
        # Build GT rows for this dt_pred and add them into the corresponding hour bucket
        rows = (
            joined[["inference_id", "y_stockout_bikes_30", "dt_plus30"]]
            .assign(y_stockout_bikes_30=lambda df: df["y_stockout_bikes_30"].astype(int))
            .itertuples(index=False, name=None)
        )

        miss_cnt = 0
        # rows is a sequence of tuples: (inference_id, label_int, dt_plus30)
        for inference_id, label_int, dtp30 in rows:
            event_id = infid_to_evid.get(str(inference_id))
            if not event_id:
                miss_cnt += 1
                continue
            y, m, d, h = _ymdh_from_utc_dt(str(dtp30))
            hour_key = f"{y}/{m}/{d}/{h}"
            hour_to_rows.setdefault(hour_key, []).append((str(inference_id), int(label_int)))
        if miss_cnt:
            print(f"[warn] {miss_cnt} records for dt={dt_pred} have no capture eventId; skipped")

    # === Write ONE jsonl per label hour (idempotent) ===
    # This avoids duplicates within the same hour and matches the folder convention:
    # s3://bucket/monitoring/ground-truth/YYYY/MM/DD/HH/labels-*.jsonl
    if not hour_to_rows:
        print("[info] No eligible dt to write labels this run (either <30m or missing actuals/preds)")
    else:
        for hour_key in sorted(hour_to_rows.keys()):
            y, m, d, h = hour_key.split("/")
            # Skip if this hour already has a labels-*.jsonl (idempotent write)
            if _gt_hour_exists(_s3(), bucket, "monitoring/ground-truth", y, m, d, h):
                print(f"[info] ground-truth already exists for hour {hour_key}; skip")
                continue

            # Compose rows for this hour and write exactly one jsonl file
            rows_for_hour = hour_to_rows[hour_key]
            # write_ground_truth_jsonl() writes to the current UTC hour by default.
            # To place it under the exact hour folder (y/m/d/h), we pass the data through a small wrapper:
            # We temporarily override "now" by computing an S3 key manually.
            # Simpler approach: reuse write_ground_truth_jsonl but trick its Body/key? -> keep simple:
            # Implement a focused writer for this case:

            # Minimal in-place writer using the exact AWS-required schema
            lines = []
            for event_id, lbl in rows_for_hour:
                rec = {
                    "groundTruthData": {"data": str(int(lbl)), "encoding": "CSV"},
                    "eventMetadata": {"eventId": str(event_id)},
                    "eventVersion": "0",
                }
                lines.append(json.dumps(rec))
            body = ("\n".join(lines) + "\n").encode("utf-8")

            gt_key = f"monitoring/ground-truth/{hour_key}/labels-{y}{m}{d}{h}{datetime.now(timezone.utc):%M%S}.jsonl"
            _s3().put_object(Bucket=bucket, Key=gt_key, Body=body, ContentType="application/json")
            print(f"[ok] wrote model-quality labels -> s3://{bucket}/{gt_key}")

    # Optional log for parquet auditing status
    if wrote_any_parquet:
        try:
            pd.read_sql("MSCK REPAIR TABLE monitoring_quality", cnx)
        except Exception:
            pass


if __name__ == "__main__":
    # PowerShell example:
    # PS> $env:SM_ENDPOINT="bikeshare-prod"; python src/inference/handler.py
    main()
