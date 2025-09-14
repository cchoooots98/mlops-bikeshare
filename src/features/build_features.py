import argparse
import json
import os

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from holidays import FR, US
from pyathena import connect
from sklearn.neighbors import BallTree

from features.schema import FEATURE_COLUMNS, LABEL_COLUMNS, validate_feature_df

EARTH_RADIUS_KM = 6371.0088


def read_env() -> dict:
    import os

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


def athena_conn(
    region: str, s3_staging_dir: str | None = None, workgroup: str = "primary", schema_name: str = "mlops_bikeshare"
):
    # PyAthena 可以使用 workgroup 默认输出；若你本地需要 staging dir，也支持传入。
    if s3_staging_dir:
        return connect(
            region_name=region,
            s3_staging_dir=s3_staging_dir,
            work_group=workgroup,
            schema_name=schema_name,
        )
    return connect(region_name=region, work_group=workgroup)


def query_df(cnx, sql: str) -> pd.DataFrame:
    return pd.read_sql(sql, cnx)


def load_status(cnx, city, start_ts, end_ts, db) -> pd.DataFrame:
    pd.read_sql("MSCK REPAIR TABLE station_status_raw", cnx)
    sql = f"""
    SELECT city, dt, station_id, bikes, docks, last_reported
    FROM {db}.v_station_status
    WHERE city='{city}'
      AND parse_datetime(dt,'yyyy-MM-dd-HH-mm')
          BETWEEN TIMESTAMP '{start_ts}:00' AND TIMESTAMP '{end_ts}:00'
    """
    return query_df(cnx, sql)


def load_latest_info(cnx, city, db) -> pd.DataFrame:
    # Read from the UNNESTed view. This view already explodes $.data.stations.
    # We take the latest dt for the given city and return one row per station.
    pd.read_sql("MSCK REPAIR TABLE station_information_raw", cnx)
    sql = f"""
    WITH latest AS (
      SELECT max(dt) AS mdt
      FROM {db}.v_station_information
      WHERE city = '{city}'
    )
    SELECT i.city,
           i.station_id,
           i.name,
           CAST(i.capacity AS integer) AS capacity,
           CAST(i.lat AS double)       AS lat,
           CAST(i.lon AS double)       AS lon
    FROM {db}.v_station_information i, latest l
    WHERE i.city = '{city}'
      AND i.dt = l.mdt
    """
    return query_df(cnx, sql)


def load_weather(cnx, city, start_ts, end_ts, db) -> pd.DataFrame:
    # Read from curated view and select all available weather fields.
    # We alias prcp_mm -> precip_mm to match downstream column names used by schema.py.
    pd.read_sql("MSCK REPAIR TABLE weather_hourly_raw", cnx)
    sql = f"""
    SELECT city,
           dt,
           ts,
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


def to_rad(x):
    return np.deg2rad(x)


def build_neighbors(info_df: pd.DataFrame, k: int = 5, max_radius_km: float = 0.8) -> pd.DataFrame:
    """
    Build a neighbor table using haversine distance (via BallTree).
    Protect against None/NaN coords and very small networks.
    """
    if info_df is None or info_df.empty:
        return pd.DataFrame(columns=["src_id", "nbr_id", "dist_km", "w"])

    # Coerce to numeric and drop invalid coords
    info = info_df.copy()
    for col in ("lat", "lon"):
        info[col] = pd.to_numeric(info[col], errors="coerce")
    info = info.dropna(subset=["lat", "lon"])
    info = info[np.isfinite(info["lat"]) & np.isfinite(info["lon"])]

    # If we have < 2 stations after cleaning, return empty
    if len(info) < 2:
        return pd.DataFrame(columns=["src_id", "nbr_id", "dist_km", "w"])

    # Degrees -> radians with guaranteed float dtype to avoid NoneType errors
    lat_rad = np.deg2rad(np.asarray(info["lat"].values, dtype="float64"))
    lon_rad = np.deg2rad(np.asarray(info["lon"].values, dtype="float64"))
    coords = np.vstack([lat_rad, lon_rad]).T

    # Build BallTree with haversine metric (expects radians)
    tree = BallTree(coords, metric="haversine")

    # Query k+1 (self + k neighbors)
    k_eff = int(min(k, max(1, len(info) - 1)))
    dist, idx = tree.query(coords, k=k_eff + 1)  # includes self at column 0
    dist_km = dist * EARTH_RADIUS_KM

    rows = []
    station_ids = info["station_id"].values
    for i in range(len(info)):
        # skip self at j=0
        for d, j in zip(dist_km[i, 1:], idx[i, 1:]):
            # respect radius if provided
            if max_radius_km and d > max_radius_km:
                continue
            rows.append((station_ids[i], station_ids[j], float(d)))

        # fallback: if nothing within radius, take closest k_eff
        if not any(r[0] == station_ids[i] for r in rows):
            for d, j in zip(dist_km[i, 1:], idx[i, 1:]):
                rows.append((station_ids[i], station_ids[j], float(d)))

    if not rows:
        return pd.DataFrame(columns=["src_id", "nbr_id", "dist_km", "w"])

    nbr = pd.DataFrame(rows, columns=["src_id", "nbr_id", "dist_km"])

    # inverse-distance weights with row normalization
    eps = 1e-6
    nbr["w"] = 1.0 / (nbr["dist_km"] + eps)
    nbr["w"] = nbr["w"] / nbr.groupby("src_id")["w"].transform("sum")
    return nbr


def _city_timezone(city: str) -> str:
    city = (city or "").lower()
    if city in ("nyc", "new_york", "new-york", "new york", "newyork"):
        return "America/New_York"
    # add more cities here if you expand the project
    return "UTC"


def align_weather_5min(weather_df, start_ts, end_ts, city="nyc") -> pd.DataFrame:
    """
    Align hourly weather (in the local time zone) to a UTC 5-minute grid,
      using merge_asof to backfill and avoid alignment holes caused by resample+reindex.
    """
    # 5分钟 UTC 目标网格
    idx5 = pd.date_range(start=start_ts, end=end_ts, freq="5min", tz="UTC")
    cols = [
        "temp_c",
        "precip_mm",
        "wind_kph",
        "rhum_pct",
        "pres_hpa",
        "wind_dir_deg",
        "wind_gust_kph",
        "snow_mm",
        "weather_code",
    ]

    if weather_df.empty:
        return pd.DataFrame({"dt": idx5.strftime("%Y-%m-%d-%H-%M"), **{c: np.nan for c in cols}})

    w = weather_df.copy()

    # 1) 解析 ts 并本地化到城市时区 → 转换到 UTC
    tz = _city_timezone(city)
    w["ts"] = pd.to_datetime(w["ts"], errors="coerce")
    w = w.dropna(subset=["ts"])
    # 注意：meteostat 的 ts 是“本地小时”，这里先本地化再转 UTC
    w["ts_utc"] = w["ts"].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="infer").dt.tz_convert("UTC")

    # 2) 只保留所需列并去重（若同一小时多条，保留最后一条）
    w = w[["ts_utc"] + cols].sort_values("ts_utc").drop_duplicates(subset=["ts_utc"], keep="last")

    # 3) 与 5min 网格做 asof 向后填充（backward），得到逐 5 分钟值
    grid = pd.DataFrame({"ts": idx5})
    joined = pd.merge_asof(
        left=grid.sort_values("ts"),
        right=w.sort_values("ts_utc"),
        left_on="ts",
        right_on="ts_utc",
        direction="backward",
        tolerance=pd.Timedelta("6h"),  # 最多向后接受 6 小时（常见逐小时数据很安全）
    )

    # 4) 生成 dt 分区键
    joined["dt"] = joined["ts"].dt.strftime("%Y-%m-%d-%H-%M")
    out = joined[["dt"] + cols].copy()

    
    # print(f"[DEBUG] weather5 asof non-null counts: {nn}")

    return out


def add_time_features(df: pd.DataFrame, city: str) -> pd.DataFrame:
    ts = pd.to_datetime(df["dt"], format="%Y-%m-%d-%H-%M", utc=True)
    df["hour"] = ts.dt.hour.astype(float)
    df["dow"] = ts.dt.dayofweek.astype(float)
    df["is_weekend"] = (df["dow"] >= 5).astype(float)
    years = list(sorted(set(ts.dt.year.tolist())))
    holiday = US(years=years) if city.lower() in {"nyc", "new_york"} else FR(years=years)
    df["is_holiday"] = ts.dt.date.map(lambda d: float(d in holiday))
    return df


def engineer(status, info, weather5, nbr, horizon_min=30, threshold=2, city="nyc") -> pd.DataFrame:
    # --- base join ---
    df = status.merge(info, on="station_id", how="left")
    if "city" not in df.columns:
        df["city"] = city

    # keep a consistent ordering and a compact dtype early
    df = df.sort_values(["station_id", "dt"]).reset_index(drop=True)
    cap = pd.to_numeric(df["capacity"], errors="coerce").replace(0, np.nan)

    df["util_bikes"] = ((df["bikes"] / cap).clip(0, 1).fillna(0.0)).astype("float32")
    df["util_docks"] = ((df["docks"] / cap).clip(0, 1).fillna(0.0)).astype("float32")

    # --- time features ---
    df = add_time_features(df, df["city"].iloc[0])

    # --- rolling features (no groupby.apply) ---
    gb = df.groupby("station_id", sort=False, observed=True)
    df["delta_bikes_5m"] = gb["bikes"].diff().fillna(0).astype("float32")
    df["delta_docks_5m"] = gb["docks"].diff().fillna(0).astype("float32")

    for win, name in [(3, "15"), (6, "30"), (12, "60")]:
        df[f"roll{name}_net_bikes"] = (
            gb["delta_bikes_5m"].rolling(win, min_periods=1).sum().reset_index(level=0, drop=True).astype("float32")
        )
        df[f"roll{name}_bikes_mean"] = (
            gb["bikes"].rolling(win, min_periods=1).mean().reset_index(level=0, drop=True).astype("float32")
        )
        df[f"roll{name}_docks_mean"] = (
            gb["docks"].rolling(win, min_periods=1).mean().reset_index(level=0, drop=True).astype("float32")
        )

    # --- neighbor aggregation (skip if no neighbors) ---
    if nbr is not None and not nbr.empty:
        neigh = df[["station_id", "dt", "bikes", "docks"]].rename(
            columns={"station_id": "nbr_id", "bikes": "nbr_bikes", "docks": "nbr_docks"}
        )
        nbr_full = (
            df[["station_id", "dt"]]
            .rename(columns={"station_id": "src_id"})
            .merge(nbr, on="src_id", how="left")
            .merge(neigh, on=["nbr_id", "dt"], how="left")
        )
        agg = (
            nbr_full.assign(wb=lambda x: x["w"] * x["nbr_bikes"], wd=lambda x: x["w"] * x["nbr_docks"])
            .groupby(["src_id", "dt"], as_index=False)[["wb", "wd"]]
            .sum()
            .rename(columns={"src_id": "station_id"})
        )
        df = df.merge(agg, on=["station_id", "dt"], how="left")
        df["nbr_bikes_weighted"] = df["wb"].fillna(0.0).astype("float32")
        df["nbr_docks_weighted"] = df["wd"].fillna(0.0).astype("float32")
        df = df.drop(columns=["wb", "wd"])
    else:
        df["nbr_bikes_weighted"] = np.float32(0.0)
        df["nbr_docks_weighted"] = np.float32(0.0)

    # --- weather join via asof ---
    ts_status = pd.to_datetime(df["dt"], format="%Y-%m-%d-%H-%M", errors="coerce", utc=True)
    df = df.assign(ts_utc=ts_status).sort_values("ts_utc")

    w = weather5.copy()
    w["ts_utc"] = pd.to_datetime(w["dt"], format="%Y-%m-%d-%H-%M", errors="coerce", utc=True)
    w = w.sort_values("ts_utc")

    tol = pd.Timedelta("15min")
    joined = pd.merge_asof(
        left=df,
        right=w.drop(columns=["dt"]),  # avoid duplicate dt col
        on="ts_utc",
        direction="backward",
        tolerance=tol,
    )

    # fixed debug print
    weather_cols = [
        "temp_c",
        "precip_mm",
        "wind_kph",
        "rhum_pct",
        "pres_hpa",
        "wind_dir_deg",
        "wind_gust_kph",
        "snow_mm",
        "weather_code",
    ]
    # hit_ratio = joined["temp_c"].notna().mean() if "temp_c" in joined.columns else 0.0
    # counts = {c: int(joined[c].notna().sum()) for c in weather_cols if c in joined.columns}
    # print(f"[DEBUG] weather merge_asof hit ratio: {hit_ratio:.1%}; non-null counts: {counts}")

    # Impute city-level weather gaps (safe because weather is city-level)
    joined = joined.sort_values("ts_utc")
    for c in weather_cols:
        if c in joined.columns:
            joined[c] = (
                joined[c]
                .astype("float32", copy=False)
                .ffill()
                .bfill()
                .fillna(0.0)  # still missing → 0
                .astype("float32", copy=False)
            )

    df = joined.drop(columns=["ts_utc"])

    # --- future targets (no groupby.apply, no reset_index copy) ---
    steps = horizon_min // 5
    gb2 = df.groupby("station_id", sort=False, observed=True)
    df["target_bikes_t30"] = gb2["bikes"].shift(-steps)
    df["target_docks_t30"] = gb2["docks"].shift(-steps)
    df["y_stockout_bikes_30"] = (df["target_bikes_t30"] <= threshold).astype("float32")
    df["y_stockout_docks_30"] = (df["target_docks_t30"] <= threshold).astype("float32")

    return df


def write_parquet_partitioned(df: pd.DataFrame, bucket: str):
    # Write local temporary files and then upload them to S3
    # (to avoid too many small files, you can first divide them into buckets by day or hour.
    # Here, we directly partition by dt)
    s3 = boto3.client("s3")
    for (city, dt_val), g in df.groupby(["city", "dt"]):
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
    ap.add_argument("--bucket", required=False)  # 默认从 env.json 取
    ap.add_argument("--start", required=False)  # 'YYYY-MM-DD HH:MM' UTC
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
    from datetime import datetime, timedelta, timezone

    end_ts = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    if args.start:
        start_ts = args.start
    else:
        end_dt = datetime.strptime(end_ts, "%Y-%m-%d %H:%M")
        start_ts = (end_dt - timedelta(days=14)).strftime("%Y-%m-%d %H:%M")

    status = load_status(cnx, args.city, start_ts, end_ts, db)

    if status.empty:
        raise RuntimeError("no status rows in the chosen window")
    n_dt = status["dt"].nunique()
    if n_dt < 100:  # 例如至少要有 ~8 小时
        raise RuntimeError(
            f"too few status snapshots in window: dt_nunique={n_dt}. " f"Check your ingestion or widen --start/--end."
        )

    info = load_latest_info(cnx, args.city, db)

    weather = load_weather(cnx, args.city, start_ts, end_ts, db)
    weather5 = align_weather_5min(weather, start_ts, end_ts, args.city)

    # print(
    #     f"[DEBUG] weather rows: {len(weather)}; weather5 rows: {len(weather5)}; "
    #     f"non-null temp_c in weather5: {weather5['temp_c'].notna().sum()}; "
    #     f"non-null temp_c in weather: {weather['temp_c'].notna().sum()}"
    # )

    bad = info[info["lat"].isna() | info["lon"].isna()]
    if not bad.empty:
        print(f"[WARN] Dropping {len(bad)} stations with missing lat/lon before neighbor build.")

    nbr = build_neighbors(info, args.neighbors, args.max_radius_km)

    df = engineer(status, info, weather5, nbr, args.horizon, args.threshold, args.city)

    # tmp = df["temp_c"].notna().mean() if "temp_c" in df.columns else 0.0
    # print(f"[DEBUG] after merge, temp_c non-null ratio: {tmp:.1%}")

    keep_cols = (
        ["city", "dt", "station_id", "name", "capacity", "lat", "lon", "bikes", "docks"]
        + FEATURE_COLUMNS
        + LABEL_COLUMNS
    )
    out = df[keep_cols].copy()

    # Drop rows that can't have labels by definition (tail after shift)
    out = out.dropna(subset=["target_bikes_t30", "target_docks_t30"])

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
            sample = out.drop(columns=["city", "dt", "station_id", "name"]).sample(
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
