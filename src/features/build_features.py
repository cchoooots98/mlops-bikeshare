import os, json, argparse
import numpy as np
import pandas as pd
from typing import Tuple
from pyathena import connect
import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from holidays import US, FR
from sklearn.neighbors import BallTree  
from schema import FEATURE_COLUMNS, LABEL_COLUMNS, validate_feature_df

EARTH_RADIUS_KM = 6371.0088

def read_env() -> dict:
    import os, json
    p = os.path.join(os.path.dirname(__file__), "..", "env.json")
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    v = raw.get("Variables", raw)  # 兼容两种结构
    # 统一小写别名，方便后续取值
    return {
        "bucket": v.get("BUCKET") or v.get("bucket"),
        "city": v.get("CITY") or v.get("city"),
        "athena_output": v.get("ATHENA_OUTPUT") or v.get("athena_output"),
        "athena_workgroup": v.get("ATHENA_WORKGROUP") or v.get("athena_workgroup") or "primary",
        "region": v.get("REGION") or v.get("region") or "ca-central-1",
        "athena_database": v.get("ATHENA_DATABASE") or v.get("athena_database") or "default",
    }


def athena_conn(region: str, s3_staging_dir: str | None = None, workgroup: str = "primary",schema_name: str= "mlops_bikeshare"):
    # PyAthena 可以使用 workgroup 默认输出；若你本地需要 staging dir，也支持传入。
    if s3_staging_dir:
        return connect(region_name=region, s3_staging_dir=s3_staging_dir, work_group=workgroup,schema_name=schema_name,)
    return connect(region_name=region, work_group=workgroup)


def query_df(cnx, sql: str) -> pd.DataFrame:
    return pd.read_sql(sql, cnx)

def load_status(cnx, city, start_ts, end_ts, db) -> pd.DataFrame:
    sql = f"""
    SELECT city, dt, station_id, bikes, docks, last_reported
    FROM {db}.v_station_status
    WHERE city='{city}'
      AND parse_datetime(dt,'yyyy-MM-dd-HH-mm')
          BETWEEN TIMESTAMP '{start_ts}:00' AND TIMESTAMP '{end_ts}:00'
    """
    return query_df(cnx, sql)

def load_latest_info(cnx, city,db) -> pd.DataFrame:
    sql = f"""
    WITH t AS (SELECT city, dt, json_row FROM {db}.station_information_raw WHERE city='{city}'),
         m AS (SELECT max(dt) AS mdt FROM t)
    SELECT
      json_extract_scalar(json_row,'$.station_id') AS station_id,
      json_extract_scalar(json_row,'$.name')       AS name,
      CAST(json_extract_scalar(json_row,'$.capacity') AS integer) AS capacity,
      CAST(json_extract_scalar(json_row,'$.lat') AS double)  AS lat,
      CAST(json_extract_scalar(json_row,'$.lon') AS double)  AS lon
    FROM t, m WHERE t.dt = m.mdt
    """
    return query_df(cnx, sql)

def load_weather(cnx, city, start_ts, end_ts, db) -> pd.DataFrame:
    # Read from curated view and select all available weather fields.
    # We alias prcp_mm -> precip_mm to match downstream column names used by schema.py.
    sql = f"""
    SELECT city,
           dt,
           temp_c,
           prcp_mm AS precip_mm,
           wind_kph,
           rhum_pct,
           pres_hpa,
           wind_dir_deg,
           wind_gust_kph,
           snow_mm,
           weather_code
    FROM {db}.v_weather_hourly
    WHERE city = '{city}'
      AND parse_datetime(dt,'yyyy-MM-dd-HH-mm')
          BETWEEN TIMESTAMP '{start_ts}:00' AND TIMESTAMP '{end_ts}:00'
    """
    return query_df(cnx, sql)


def to_rad(x): return np.deg2rad(x)

def build_neighbors(info_df, k=5, max_radius_km=0.8) -> pd.DataFrame:
    coords = np.vstack([to_rad(info_df["lat"].values), to_rad(info_df["lon"].values)]).T
    tree = BallTree(coords, metric="haversine")
    dist, idx = tree.query(coords, k=k+1)  # 包含自身
    dist_km = dist * EARTH_RADIUS_KM
    pairs = []
    for i, (di, ii) in enumerate(zip(dist_km, idx)):
        for d, j in zip(di[1:], ii[1:]):  # 去自身
            if d <= max_radius_km:
                pairs.append((info_df.iloc[i]["station_id"], info_df.iloc[j]["station_id"], d))
    if not pairs:
        # 回退：没命中半径，用最近 k 个
        for i, (di, ii) in enumerate(zip(dist_km, idx)):
            for d, j in zip(di[1:k+1], ii[1:k+1]):
                pairs.append((info_df.iloc[i]["station_id"], info_df.iloc[j]["station_id"], d))
    nbr = pd.DataFrame(pairs, columns=["src_id","nbr_id","dist_km"])
    nbr["w"] = 1.0 / (nbr["dist_km"] + 1e-6)
    nbr["w"] = nbr["w"] / nbr.groupby("src_id")["w"].transform("sum")
    return nbr

def align_weather_5min(weather_df, start_ts, end_ts) -> pd.DataFrame:
    # Align hourly weather to 5-minute grid by forward-fill.
    if weather_df.empty:
        idx5 = pd.date_range(start=start_ts, end=end_ts, freq="5min", tz="UTC")
        # Build an empty skeleton for all weather columns we expect downstream
        return pd.DataFrame({
            "dt": idx5.strftime("%Y-%m-%d-%H-%M"),
            "temp_c": np.nan,
            "precip_mm": np.nan,
            "wind_kph": np.nan,
            "rhum_pct": np.nan,
            "pres_hpa": np.nan,
            "wind_dir_deg": np.nan,
            "wind_gust_kph": np.nan,
            "snow_mm": np.nan,
            "weather_code": np.nan,
        })

    w = weather_df.copy()
    # Input dt is partition string 'YYYY-MM-DD-HH-MM'; we re-create a timestamp index
    w["ts"] = pd.to_datetime(w["dt"], format="%Y-%m-%d-%H-%M", utc=True)

    # Reindex only the columns we want to propagate; forward fill to 5-minute grid
    cols = [
        "temp_c", "precip_mm", "wind_kph",
        "rhum_pct", "pres_hpa", "wind_dir_deg",
        "wind_gust_kph", "snow_mm", "weather_code"
    ]
    w = (
        w.set_index("ts")
         .sort_index()
         .reindex(columns=cols)
         .resample("5min").ffill()
         .reset_index()
    )
    w["dt"] = w["ts"].dt.strftime("%Y-%m-%d-%H-%M")
    return w.drop(columns=["ts"])


def add_time_features(df: pd.DataFrame, city: str) -> pd.DataFrame:
    ts = pd.to_datetime(df["dt"], format="%Y-%m-%d-%H-%M", utc=True)
    df["hour"] = ts.dt.hour.astype(float)
    df["dow"] = ts.dt.dayofweek.astype(float)
    df["is_weekend"] = (df["dow"] >= 5).astype(float)
    years = list(sorted(set(ts.dt.year.tolist())))
    holiday = US(years=years) if city.lower() in {"nyc","new_york"} else FR(years=years)
    df["is_holiday"] = ts.dt.date.map(lambda d: float(d in holiday))
    return df

def engineer(status, info, weather5, nbr, horizon_min=30, threshold=2) -> pd.DataFrame:
    df = status.merge(info, on="station_id", how="left")
    df["util_bikes"] = (df["bikes"] / df["capacity"].replace(0, np.nan)).clip(0,1).fillna(0.0)
    df["util_docks"] = (df["docks"] / df["capacity"].replace(0, np.nan)).clip(0,1).fillna(0.0)
    df = add_time_features(df, df["city"].iloc[0])

    df = df.sort_values(["station_id","dt"])

    def _rolling(g: pd.DataFrame) -> pd.DataFrame:
        g["delta_bikes_5m"] = g["bikes"].diff().fillna(0)
        g["delta_docks_5m"] = g["docks"].diff().fillna(0)
        for win, name in [(3,"15"),(6,"30"),(12,"60")]:
            g[f"roll{name}_net_bikes"] = g["delta_bikes_5m"].rolling(win, min_periods=1).sum()
            g[f"roll{name}_bikes_mean"] = g["bikes"].rolling(win, min_periods=1).mean()
            g[f"roll{name}_docks_mean"] = g["docks"].rolling(win, min_periods=1).mean()
        return g
    df = df.groupby("station_id", group_keys=False).apply(_rolling)

    # 邻域聚合（按时间对齐）
    neigh = df[["station_id","dt","bikes","docks"]]\
            .rename(columns={"station_id":"nbr_id","bikes":"nbr_bikes","docks":"nbr_docks"})
    nbr_full = (df[["station_id","dt"]].rename(columns={"station_id":"src_id"})
                .merge(nbr.merge(neigh, on="nbr_id", how="left"), on=["src_id","dt"], how="left"))
    agg = (nbr_full
           .assign(wb=lambda x: x["w"]*x["nbr_bikes"], wd=lambda x: x["w"]*x["nbr_docks"])
           .groupby(["src_id","dt"], as_index=False)[["wb","wd"]].sum()
           .rename(columns={"src_id":"station_id"}))
    df = df.merge(agg, on=["station_id","dt"], how="left")
    df["nbr_bikes_weighted"] = df["wb"].fillna(0.0)
    df["nbr_docks_weighted"] = df["wd"].fillna(0.0)
    df = df.drop(columns=["wb","wd"])

    # 天气（5min 对齐）
    df = df.merge(weather5, on="dt", how="left")

    # 生成 t+30 标签与回归目标（严格只 shift 目标，避免时间泄漏）
    steps = horizon_min // 5
    def _future(g: pd.DataFrame) -> pd.DataFrame:
        g["target_bikes_t30"] = g["bikes"].shift(-steps)
        g["target_docks_t30"] = g["docks"].shift(-steps)
        g["y_stockout_bikes_30"] = (g["target_bikes_t30"] <= threshold).astype(float)
        g["y_stockout_docks_30"] = (g["target_docks_t30"] <= threshold).astype(float)
        return g
    df = df.groupby("station_id", group_keys=False).apply(_future)
    return df

def write_parquet_partitioned(df: pd.DataFrame, bucket: str):
    # 写本地临时文件再上传 S3（避免小文件过多可先按天/小时分桶，这里按 dt 直接分区）
    s3 = boto3.client("s3")
    for (city, dt_val), g in df.groupby(["city","dt"]):
        table_path = f"features/city={city}/dt={dt_val}/part-0.parquet"
        table_local = f"/tmp/{city}_{dt_val}.parquet"
        table_local_dir = os.path.dirname(table_local)
        os.makedirs(table_local_dir, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(g), table_local)
        s3.upload_file(table_local, bucket, table_path)

def create_table_if_not_exists(cnx, bucket: str):
    # Create the external table only once (or when schema changes).
    # IMPORTANT: Do NOT repeat partition columns in the regular column list.
    sql = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS features_offline (
    station_id string,
    name string,
    capacity int,
    lat double,
    lon double,
    bikes int,
    docks int,
    util_bikes double,
    util_docks double,
    delta_bikes_5m double,
    delta_docks_5m double,
    roll15_net_bikes double,
    roll30_net_bikes double,
    roll60_net_bikes double,
    roll15_bikes_mean double,
    roll30_bikes_mean double,
    roll60_bikes_mean double,
    nbr_bikes_weighted double,
    nbr_docks_weighted double,
    hour double,
    dow double,
    is_weekend double,
    is_holiday double,
    temp_c double,
    precip_mm double,
    wind_kph double,
    rhum_pct double,
    pres_hpa double,
    wind_dir_deg double,
    wind_gust_kph double,
    snow_mm double,
    weather_code double,
    y_stockout_bikes_30 double,
    y_stockout_docks_30 double,
    target_bikes_t30 double,
    target_docks_t30 double
    )
    PARTITIONED BY (city string, dt string)
    STORED AS PARQUET
    LOCATION 's3://{bucket}/features/'
    TBLPROPERTIES ('parquet.compression'='SNAPPY');
    """
    pd.read_sql(sql, cnx)
    pd.read_sql("MSCK REPAIR TABLE features_offline", cnx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--region", default="ca-central-1")
    ap.add_argument("--bucket", required=False)      # 默认从 env.json 取
    ap.add_argument("--start", required=False)       # 'YYYY-MM-DD HH:MM' UTC
    ap.add_argument("--end", required=False)
    ap.add_argument("--neighbors", type=int, default=5)
    ap.add_argument("--max-radius-km", type=float, default=0.8)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--threshold", type=int, default=2)
    ap.add_argument("--eda", action="store_true")
    args = ap.parse_args()

    cfg = read_env()
    bucket = args.bucket or cfg["bucket"]
    region = args.region or cfg["region"]
    
    cnx = athena_conn(
        region=region,
        s3_staging_dir=cfg["athena_output"],
        workgroup=cfg["athena_workgroup"],
        schema_name=cfg["athena_database"],
    )

    db = cfg["athena_database"]


    # 默认最近 14 天
    from datetime import datetime, timezone, timedelta
    end_ts = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    if args.start:
        start_ts = args.start
    else:
        end_dt = datetime.strptime(end_ts, "%Y-%m-%d %H:%M")
        start_ts = (end_dt - timedelta(days=14)).strftime("%Y-%m-%d %H:%M")

    status = load_status(cnx, args.city, start_ts, end_ts,db)
    if status.empty:
        raise RuntimeError("no status rows in the chosen window")
    info = load_latest_info(cnx, args.city,db)
    weather = load_weather(cnx, args.city, start_ts, end_ts,db)
    weather5 = align_weather_5min(weather, start_ts, end_ts)
    nbr = build_neighbors(info, args.neighbors, args.max_radius_km)

    df = engineer(status, info, weather5, nbr, args.horizon, args.threshold)

    keep_cols = (["city","dt","station_id","name","capacity","lat","lon","bikes","docks"]
                 + FEATURE_COLUMNS + LABEL_COLUMNS)
    out = df[keep_cols].copy()

    # 质量校验（延续你 validators 的风格）
    validate_feature_df(out)

    # 写 S3（分区 city/dt）
    write_parquet_partitioned(out, bucket)

    # Athena 建/修表（只需首次或 schema 变化时执行一次）
    try:
        create_table_if_not_exists(cnx, bucket)
    except Exception:
        # 表已存在则做修复
        pd.read_sql("MSCK REPAIR TABLE features_offline", cnx)

    # 可选：简版 EDA 报告
    if args.eda:
        try:
            from ydata_profiling import ProfileReport
            os.makedirs("build/reports", exist_ok=True)
            sample = out.drop(columns=["city","dt","station_id","name"]).sample(
                min(20000, len(out)), random_state=42
            )
            rep = ProfileReport(sample, title=f"Features EDA — {args.city}", minimal=True, explorative=True)
            from datetime import datetime, timezone
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            local_path = f"build/reports/{args.city}_{ts}.html"
            rep.to_file(local_path)
            # 上传 S3
            boto3.client("s3").upload_file(local_path, bucket, f"features/reports/{args.city}/{ts}/index.html")
            print("EDA report uploaded to s3://%s/features/reports/%s/%s/index.html" % (bucket, args.city, ts))
        except Exception as e:
            print("skip EDA:", e)

if __name__ == "__main__":
    main()
