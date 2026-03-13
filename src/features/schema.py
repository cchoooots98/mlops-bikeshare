import re
from typing import Iterable, Sequence

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
DEFAULT_PER_COLUMN_MISSING_RATE_THRESHOLD = 0.10
DEFAULT_STATION_INVENTORY_CAPACITY_MULTIPLIER = 2.0

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


def _raise_for_integer_range(
    df: pd.DataFrame,
    columns: Iterable[str],
    min_value: int,
    max_value: int,
    *,
    allow_null: bool = False,
) -> None:
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce")
        integer_mask = series.isna() | (series == series.round())
        range_mask = series.between(min_value, max_value, inclusive="both")
        mask = integer_mask & range_mask
        if allow_null:
            mask = mask | series.isna()
        if not mask.all():
            bad = int((~mask).sum())
            raise ValueError(
                f"{column} must be an integer between {min_value} and {max_value}; found {bad} invalid values"
            )


def _raise_for_training_outputs(df: pd.DataFrame) -> None:
    _raise_for_binary(df, ["y_stockout_bikes_30", "y_stockout_docks_30"], allow_null=True)
    _raise_for_non_negative(df, ["target_bikes_t30", "target_docks_t30"], allow_null=True)


def _raise_for_station_inventory_limit(
    df: pd.DataFrame, *, capacity_multiplier: float = DEFAULT_STATION_INVENTORY_CAPACITY_MULTIPLIER
) -> None:
    bikes = pd.to_numeric(df["bikes"], errors="coerce")
    docks = pd.to_numeric(df["docks"], errors="coerce")
    capacity = pd.to_numeric(df["capacity"], errors="coerce")
    mask = (bikes + docks) <= (capacity * capacity_multiplier)
    if not mask.all():
        bad = int((~mask).sum())
        raise ValueError(
            "bikes + docks must be within configured station capacity tolerance; "
            f"found {bad} invalid values"
        )


def _raise_for_unique_entity_grain(df: pd.DataFrame) -> None:
    duplicates = df.duplicated(subset=ENTITY_COLUMNS, keep=False)
    if duplicates.any():
        bad = int(duplicates.sum())
        raise ValueError(f"entity grain city+dt+station_id must be unique; found {bad} duplicate rows")


def _resolve_feature_columns(feature_columns: Sequence[str] | None) -> list[str]:
    columns = list(feature_columns or FEATURE_COLUMNS)
    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        raise ValueError(f"feature columns must be unique; duplicates={duplicates}")
    return columns


def _raise_for_all_nan_features(df: pd.DataFrame, non_nullable_feature_columns: Sequence[str]) -> None:
    all_nan = [
        column
        for column in non_nullable_feature_columns
        if df[column].isna().all()
    ]
    if all_nan:
        raise ValueError(f"non-nullable feature columns are entirely null: {all_nan}")


def _raise_for_per_column_missing_rate(
    df: pd.DataFrame,
    threshold: float,
    non_nullable_feature_columns: Sequence[str],
) -> None:
    if not non_nullable_feature_columns:
        return
    missing_rates = df[list(non_nullable_feature_columns)].isna().mean()
    offenders = sorted(
        f"{column}={rate:.2%}"
        for column, rate in missing_rates.items()
        if float(rate) > threshold
    )
    if offenders:
        raise ValueError(
            "feature per-column missing rate exceeded threshold "
            f"{threshold:.2%}: {offenders}"
        )


def validate_feature_df(
    df: pd.DataFrame,
    *,
    require_labels: bool = True,
    feature_columns: Sequence[str] | None = None,
    missing_rate_threshold: float = DEFAULT_MISSING_RATE_THRESHOLD,
    per_column_missing_rate_threshold: float = DEFAULT_PER_COLUMN_MISSING_RATE_THRESHOLD,
    station_inventory_capacity_multiplier: float = DEFAULT_STATION_INVENTORY_CAPACITY_MULTIPLIER,
) -> bool:
    selected_feature_columns = _resolve_feature_columns(feature_columns)
    nullable_feature_columns = [column for column in selected_feature_columns if column in NULLABLE_FEATURE_COLUMNS]
    non_nullable_feature_columns = [
        column for column in selected_feature_columns if column not in nullable_feature_columns
    ]
    expected_columns = ENTITY_COLUMNS + REQUIRED_BASE + selected_feature_columns
    if require_labels:
        expected_columns = expected_columns + LABEL_COLUMNS

    _raise_for_missing_columns(df, expected_columns)
    _raise_for_forbidden_columns(df)

    if df.empty:
        return True

    _raise_for_invalid_dt(df)
    _raise_for_unique_entity_grain(df)
    _raise_for_non_negative(df, ["capacity", "bikes", "docks"], allow_null=False)
    _raise_for_station_inventory_limit(
        df,
        capacity_multiplier=station_inventory_capacity_multiplier,
    )
    _raise_for_range(df, ["lat"], -90.0, 90.0, allow_null=False)
    _raise_for_range(df, ["lon"], -180.0, 180.0, allow_null=False)
    _raise_for_non_negative(
        df,
        [
            column
            for column in ("minutes_since_prev_snapshot", "neighbor_count_within_radius")
            if column in selected_feature_columns
        ],
        allow_null=False,
    )
    _raise_for_integer_range(
        df,
        [column for column in ("neighbor_count_within_radius",) if column in selected_feature_columns],
        0,
        10000,
        allow_null=False,
    )
    _raise_for_range(
        df,
        [column for column in ("util_bikes", "util_docks") if column in selected_feature_columns],
        0.0,
        station_inventory_capacity_multiplier,
        allow_null=False,
    )
    _raise_for_binary(
        df,
        [column for column in ("has_neighbors_within_radius", "is_weekend", "is_holiday") if column in selected_feature_columns],
        allow_null=False,
    )
    _raise_for_integer_range(
        df,
        [column for column in ("hour",) if column in selected_feature_columns],
        0,
        23,
        allow_null=False,
    )
    _raise_for_integer_range(
        df,
        [column for column in ("dow",) if column in selected_feature_columns],
        1,
        7,
        allow_null=False,
    )
    _raise_for_integer_range(
        df,
        [column for column in ("weather_code", "hourly_weather_code") if column in selected_feature_columns],
        0,
        9999,
        allow_null=True,
    )
    _raise_for_range(
        df,
        [column for column in ("humidity_pct", "hourly_precipitation_probability_pct") if column in selected_feature_columns],
        0.0,
        100.0,
        allow_null=True,
    )
    if require_labels:
        _raise_for_training_outputs(df)
    _raise_for_all_nan_features(df, non_nullable_feature_columns)
    _raise_for_per_column_missing_rate(df, per_column_missing_rate_threshold, non_nullable_feature_columns)

    miss = 0.0
    if non_nullable_feature_columns:
        miss = float(df[list(non_nullable_feature_columns)].isna().mean().mean())
    if miss > missing_rate_threshold:
        raise ValueError(f"feature missing rate {miss:.2%} > {missing_rate_threshold:.2%}")

    return True
