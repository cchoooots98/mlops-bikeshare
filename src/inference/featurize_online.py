# featurize_online.py
# Windows/VSCode friendly. All comments in English.
# Goal: Build ONLINE features for the latest 5-min snapshot, using the SAME logic as offline.
# We *import* the same functions/columns from your offline code to guarantee parity.

from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# # Ensure we can import "features.*" if script is executed directly from repo root
# REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if REPO_ROOT not in sys.path:
#     sys.path.insert(0, REPO_ROOT)
from src.features.build_features import (  # reused to avoid skew
    align_weather_5min,
    athena_conn,
    build_neighbors,
    engineer,
    load_latest_info,
    load_status,
    load_weather,
    read_env,
)
from src.features.schema import FEATURE_COLUMNS  # exact same feature list as training


def _latest_dt_ts(cnx, city: str, db: str) -> str:
    """
    Query the latest status snapshot timestamp for CITY from v_station_status.
    We return a '"%Y-%m-%d %H:%M"' string, which matches your 'dt' partition key.
    """
    sql = f"""
    SELECT max(dt_ts) AS latest_ts
    FROM {db}.v_station_status
    WHERE city = '{city}'
    """

    df = pd.read_sql(sql, cnx)
    if df.empty or df.loc[0, "latest_ts"] is None:
        raise RuntimeError("No station status found for city.")
    ts = pd.to_datetime(df.loc[0, "latest_ts"], utc=True)

    return ts.strftime("%Y-%m-%d %H:%M")


def build_online_features(city: str) -> pd.DataFrame:
    """
    Build features for the *latest* 5-minute snapshot.
    To compute rolling features (15/30/60 minutes), we read the last ~65 minutes of status.
    This exactly mirrors the rolling windows used offline via 'engineer(...)'.
    """
    cfg = read_env()  # same env loader as offline
    cnx = athena_conn(
        region=cfg["region"],
        s3_staging_dir=cfg["athena_output"],
        workgroup=cfg["athena_workgroup"],
        schema_name=cfg["athena_database"],
    )
    db = cfg["athena_database"]

    # 1) Find latest dt for CITY
    latest_dt = _latest_dt_ts(cnx, city, db)  # (a string)

    # Window start is ~65 minutes before the latest to cover 12 steps (60m) + a safety margin

    end_ts = datetime.strptime(latest_dt, "%Y-%m-%d %H:%M")  # datetime object

    start_ts = (end_ts - timedelta(minutes=65)).strftime("%Y-%m-%d %H:%M")

    # 2) Load recent status window (same source views as offline)
    # sql_status = f"""
    #     SELECT city, dt, dt_ts, station_id, bikes, docks, last_reported
    #     FROM {db}.v_station_status
    #     WHERE city = '{city}'
    #       AND parse_datetime(dt,'%Y-%m-%d %H:%M')
    #           BETWEEN TIMESTAMP '{start_ts}:00' AND TIMESTAMP '{latest_dt}:00'
    # """
    ########
    status = load_status(cnx, city, start_ts, latest_dt, db)
    # print(f"[DEBUG_online] the shape of the status:{status.shape}")
    # status = query_df(cnx, sql_status)
    if status.empty:
        raise RuntimeError("No recent status rows in the chosen window.")

    # 3) Latest station information (same method as offline)
    info = load_latest_info(cnx, city, db)

    # 4) Weather: load hourlies, align to 5-min grid, and merge with status using the SAME merge_asof rules
    weather = load_weather(cnx, city, start_ts, latest_dt, db)
    # print(f"[DEBUG_online] the shape of the weather: {weather.shape}")

    weather5 = align_weather_5min(weather, start_ts, latest_dt, city)
    # print(f"[DEBUG_online] the shape of the weather5: {weather5.shape}")

    # 5) Neighbor table (same BallTree haversine approach as offline)
    nbr = build_neighbors(info, k=5, max_radius_km=0.8)

    # 6) Engineer features with THE SAME function used offline:
    #    - rolling deltas/means for 15/30/60 minutes
    #    - neighbor weighted bikes/docks
    #    - time & holiday features
    df_full = engineer(
        status=status,
        info=info,
        weather5=weather5,
        nbr=nbr,
        horizon_min=30,  # horizon is irrelevant for online features; labels are dropped below
        threshold=2,
        city=city,
    )

    # 1) Parse df_full["dt"] into datetime64[ns, UTC]
    # dt format looks like "2025-09-12-23-05"
    df_full["ts_utc"] = pd.to_datetime(
        df_full["dt"], format="%Y-%m-%d-%H-%M", errors="coerce", utc=True  # must match your dt format exactly
    )

    # 2) Normalize end_ts to a pandas.Timestamp with UTC timezone
    # If end_ts is a string like "2025-09-13-00-10":
    #   end_ts = pd.to_datetime(end_ts, format="%Y-%m-%d-%H-%M", utc=True)
    # If it's already datetime or Timestamp, unify it to UTC:
    end_ts = (
        pd.to_datetime(end_ts, utc=True)
        if getattr(end_ts, "tzinfo", None) is None
        else pd.to_datetime(end_ts).tz_convert("UTC")
    )

    # Align to the nearest minute to avoid seconds/milliseconds mismatches
    end_ts = end_ts.floor("min")

    # 3) Try exact match first
    latest = df_full.loc[df_full["ts_utc"] == end_ts].copy()

    latest = latest.drop(columns=["ts_utc"])

    # Minimal sanity checks to protect serving:
    if latest.empty:
        raise RuntimeError(f"No rows for latest dt={end_ts}.")
    for c in FEATURE_COLUMNS:
        if c not in latest.columns:
            raise RuntimeError(f"Missing expected feature column: {c}")

    # Assemble the feature matrix for the model
    X = latest[["city", "dt", "station_id"] + FEATURE_COLUMNS].copy()
    X[FEATURE_COLUMNS] = X[FEATURE_COLUMNS].astype(np.float32)
    return X


if __name__ == "__main__":
    # CLI for ad-hoc testing:
    # PS> python src/inference/featurize_online.py
    cfg = read_env()
    city = cfg["city"]
    out = build_online_features(city)
    # Print a small preview for debugging in VSCode terminal
    print(out.head(5).to_string(index=False))
