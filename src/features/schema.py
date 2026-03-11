import re
from typing import Iterable

import pandas as pd

# dbt feature tables are the formal producer contract; Python validates and consumes that contract.
ENTITY_COLUMNS = ["city", "dt", "station_id"]

REQUIRED_BASE = ["capacity", "lat", "lon", "bikes", "docks"]

FEATURE_COLUMNS = [
    "minutes_since_prev_snapshot",
    "util_bikes",
    "util_docks",
    "delta_bikes_5m",
    "delta_docks_5m",
    "roll15_net_bikes",
    "roll30_net_bikes",
    "roll60_net_bikes",
    "roll15_bikes_mean",
    "roll30_bikes_mean",
    "roll60_bikes_mean",
    "nbr_bikes_weighted",
    "nbr_docks_weighted",
    "has_neighbors_within_radius",
    "neighbor_count_within_radius",
    "hour",
    "dow",
    "is_weekend",
    "is_holiday",
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms",
    "precipitation_mm",
    "weather_code",
    "hourly_temperature_c",
    "hourly_humidity_pct",
    "hourly_wind_speed_ms",
    "hourly_precipitation_mm",
    "hourly_precipitation_probability_pct",
    "hourly_weather_code",
]

LABEL_COLUMNS = ["y_stockout_bikes_30", "y_stockout_docks_30", "target_bikes_t30", "target_docks_t30"]

ONLINE_REQUIRED_COLUMNS = ENTITY_COLUMNS + REQUIRED_BASE + FEATURE_COLUMNS
TRAINING_REQUIRED_COLUMNS = ONLINE_REQUIRED_COLUMNS + LABEL_COLUMNS

WEATHER_FEATURE_COLUMNS = [
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms",
    "precipitation_mm",
    "weather_code",
    "hourly_temperature_c",
    "hourly_humidity_pct",
    "hourly_wind_speed_ms",
    "hourly_precipitation_mm",
    "hourly_precipitation_probability_pct",
    "hourly_weather_code",
]

NULLABLE_FEATURE_COLUMNS = ["nbr_bikes_weighted", "nbr_docks_weighted"] + WEATHER_FEATURE_COLUMNS
NON_NULLABLE_FEATURE_COLUMNS = [column for column in FEATURE_COLUMNS if column not in NULLABLE_FEATURE_COLUMNS]

# Previous threshold was 1% across all 30 features. After excluding nullable-by-contract
# weather/neighbor columns, 2% across the remaining 17 columns preserves similar strictness.
DEFAULT_MISSING_RATE_THRESHOLD = 0.02

FORBIDDEN_LEGACY_COLUMNS = [
    "weather_main",
    "hourly_weather_main",
    "weather_description",
    "hourly_forecast_at",
    "source",
    "snapshot_bucket_at_utc",
    "temp_c",
    "precip_mm",
    "wind_kph",
    "rhum_pct",
    "pres_hpa",
    "wind_dir_deg",
    "wind_gust_kph",
    "snow_mm",
]

_DT_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}$")


def _raise_for_missing_columns(df: pd.DataFrame, expected_columns: Iterable[str]) -> None:
    missing = [column for column in expected_columns if column not in df.columns]
    if missing:
        raise ValueError(f"missing expected columns: {missing}")


def _raise_for_forbidden_columns(df: pd.DataFrame) -> None:
    forbidden = sorted(set(df.columns).intersection(FORBIDDEN_LEGACY_COLUMNS))
    if forbidden:
        raise ValueError(f"forbidden legacy columns present: {forbidden}")


def _raise_for_invalid_dt(df: pd.DataFrame) -> None:
    dt_values = df["dt"].astype("string")
    valid_mask = dt_values.notna() & dt_values.str.match(_DT_PATTERN)
    if not valid_mask.all():
        bad = int((~valid_mask).sum())
        raise ValueError(f"dt must match YYYY-MM-DD-HH-mm; found {bad} invalid values")


def _raise_for_non_negative(df: pd.DataFrame, columns: Iterable[str], allow_null: bool = False) -> None:
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce")
        mask = (series >= 0) if not allow_null else (series.isna() | (series >= 0))
        if not mask.all():
            bad = int((~mask).sum())
            raise ValueError(f"{column} must be non-negative; found {bad} invalid values")


def _raise_for_range(
    df: pd.DataFrame, columns: Iterable[str], min_value: float, max_value: float, allow_null: bool = True
) -> None:
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce")
        mask = series.between(min_value, max_value, inclusive="both")
        if allow_null:
            mask = mask | series.isna()
        if not mask.all():
            bad = int((~mask).sum())
            raise ValueError(f"{column} must be between {min_value} and {max_value}; found {bad} invalid values")


def _raise_for_binary(df: pd.DataFrame, columns: Iterable[str], allow_null: bool = False) -> None:
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce")
        mask = series.isin([0, 1])
        if allow_null:
            mask = mask | series.isna()
        if not mask.all():
            bad = int((~mask).sum())
            raise ValueError(f"{column} must be 0/1; found {bad} invalid values")


def _raise_for_all_nan_features(df: pd.DataFrame) -> None:
    all_nan = [
        column
        for column in FEATURE_COLUMNS
        if column not in NULLABLE_FEATURE_COLUMNS and df[column].isna().all()
    ]
    if all_nan:
        raise ValueError(f"non-nullable feature columns are entirely null: {all_nan}")


def validate_feature_df(
    df: pd.DataFrame, *, require_labels: bool = True, missing_rate_threshold: float = DEFAULT_MISSING_RATE_THRESHOLD
) -> bool:
    expected_columns = TRAINING_REQUIRED_COLUMNS if require_labels else ONLINE_REQUIRED_COLUMNS

    _raise_for_missing_columns(df, expected_columns)
    _raise_for_forbidden_columns(df)

    if df.empty:
        return True

    _raise_for_invalid_dt(df)
    _raise_for_non_negative(df, ["capacity", "bikes", "docks"], allow_null=False)
    _raise_for_range(df, ["lat"], -90.0, 90.0, allow_null=False)
    _raise_for_range(df, ["lon"], -180.0, 180.0, allow_null=False)
    _raise_for_non_negative(df, ["minutes_since_prev_snapshot", "neighbor_count_within_radius"], allow_null=False)
    _raise_for_range(df, ["util_bikes", "util_docks"], 0.0, 1.0, allow_null=False)
    _raise_for_binary(df, ["has_neighbors_within_radius", "is_weekend", "is_holiday"], allow_null=False)
    _raise_for_range(df, ["humidity_pct", "hourly_precipitation_probability_pct"], 0.0, 100.0, allow_null=True)
    _raise_for_all_nan_features(df)

    miss = df[NON_NULLABLE_FEATURE_COLUMNS].isna().mean().mean()
    if miss > missing_rate_threshold:
        raise ValueError(f"feature missing rate {miss:.2%} > {missing_rate_threshold:.2%}")

    return True
