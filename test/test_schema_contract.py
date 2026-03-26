import pandas as pd
import pytest
from src.features.schema import (
    DEFAULT_MISSING_RATE_THRESHOLD,
    DEFAULT_PER_COLUMN_MISSING_RATE_THRESHOLD,
    DEFAULT_STATION_INVENTORY_CAPACITY_MULTIPLIER,
    ENTITY_COLUMNS,
    FEATURE_COLUMNS,
    FORBIDDEN_LEGACY_COLUMNS,
    LABEL_COLUMNS,
    NON_NULLABLE_FEATURE_COLUMNS,
    NULLABLE_FEATURE_COLUMNS,
    ONLINE_REQUIRED_COLUMNS,
    REQUIRED_BASE,
    TRAINING_REQUIRED_COLUMNS,
    WEATHER_FEATURE_COLUMNS,
    validate_feature_df,
)


def _valid_training_df() -> pd.DataFrame:
    row = {
        "city": "paris",
        "dt": "2026-03-11-10-05",
        "station_id": "station-001",
        "capacity": 40,
        "lat": 48.8566,
        "lon": 2.3522,
        "bikes": 12,
        "docks": 28,
        "minutes_since_prev_snapshot": 5.0,
        "util_bikes": 0.30,
        "util_docks": 0.70,
        "delta_bikes_5m": 1.0,
        "delta_docks_5m": -1.0,
        "roll15_net_bikes": 2.0,
        "roll30_net_bikes": 3.0,
        "roll60_net_bikes": 4.0,
        "roll15_bikes_mean": 10.0,
        "roll30_bikes_mean": 11.0,
        "roll60_bikes_mean": 12.0,
        "nbr_bikes_weighted": 9.5,
        "nbr_docks_weighted": 18.5,
        "has_neighbors_within_radius": 1,
        "neighbor_count_within_radius": 4,
        "hour": 10,
        "dow": 2,
        "is_weekend": 0,
        "is_holiday": 0,
        "temperature_c": 16.2,
        "humidity_pct": 55.0,
        "wind_speed_ms": 4.8,
        "precipitation_mm": 0.2,
        "weather_code": 3,
        "hourly_temperature_c": 16.0,
        "hourly_humidity_pct": 56.0,
        "hourly_wind_speed_ms": 5.1,
        "hourly_precipitation_mm": 0.3,
        "hourly_precipitation_probability_pct": 35.0,
        "hourly_weather_code": 3,
        "y_stockout_bikes_30": 0,
        "y_stockout_docks_30": 0,
        "target_bikes_t30": 10,
        "target_docks_t30": 30,
    }
    return pd.DataFrame([row], columns=TRAINING_REQUIRED_COLUMNS)


def test_feature_columns_match_dbt_contract_order():
    assert FEATURE_COLUMNS == [
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


def test_entity_required_base_and_label_columns_match_contract():
    assert ENTITY_COLUMNS == ["city", "dt", "station_id"]
    assert REQUIRED_BASE == ["capacity", "lat", "lon", "bikes", "docks"]
    assert LABEL_COLUMNS == ["y_stockout_bikes_30", "y_stockout_docks_30", "target_bikes_t30", "target_docks_t30"]
    assert ONLINE_REQUIRED_COLUMNS == ENTITY_COLUMNS + REQUIRED_BASE + FEATURE_COLUMNS
    assert TRAINING_REQUIRED_COLUMNS == ONLINE_REQUIRED_COLUMNS + LABEL_COLUMNS
    assert DEFAULT_MISSING_RATE_THRESHOLD == 0.02
    assert DEFAULT_PER_COLUMN_MISSING_RATE_THRESHOLD == 0.10
    assert DEFAULT_STATION_INVENTORY_CAPACITY_MULTIPLIER == 2.0
    assert NON_NULLABLE_FEATURE_COLUMNS == [
        column for column in FEATURE_COLUMNS if column not in NULLABLE_FEATURE_COLUMNS
    ]


def test_validate_feature_df_rejects_forbidden_legacy_columns():
    df = _valid_training_df()
    df[FORBIDDEN_LEGACY_COLUMNS[0]] = "legacy"

    with pytest.raises(ValueError, match="forbidden legacy columns present"):
        validate_feature_df(df)


def test_validate_feature_df_requires_labels_for_training_only():
    online_df = _valid_training_df()[ONLINE_REQUIRED_COLUMNS].copy()

    assert validate_feature_df(online_df, require_labels=False) is True

    with pytest.raises(ValueError, match="missing expected columns"):
        validate_feature_df(online_df, require_labels=True)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("minutes_since_prev_snapshot", -1.0),
        ("neighbor_count_within_radius", -1),
    ],
)
def test_validate_feature_df_rejects_negative_contract_values(column, value):
    df = _valid_training_df()
    df.loc[0, column] = value

    with pytest.raises(ValueError, match="must be non-negative"):
        validate_feature_df(df)


def test_validate_feature_df_allows_utilization_above_one_within_capacity_tolerance():
    df = _valid_training_df()
    df.loc[0, "bikes"] = 60
    df.loc[0, "docks"] = 20
    df.loc[0, "util_bikes"] = 1.5
    df.loc[0, "util_docks"] = 0.5

    assert validate_feature_df(df) is True


def test_validate_feature_df_rejects_inventory_above_capacity_tolerance():
    df = _valid_training_df()
    df.loc[0, "bikes"] = 61
    df.loc[0, "docks"] = 20
    df.loc[0, "util_bikes"] = 1.525
    df.loc[0, "util_docks"] = 0.5

    with pytest.raises(ValueError, match="station capacity tolerance"):
        validate_feature_df(df)


def test_validate_feature_df_rejects_utilization_above_capacity_tolerance():
    df = _valid_training_df()
    df.loc[0, "bikes"] = 60
    df.loc[0, "docks"] = 20
    df.loc[0, "util_bikes"] = 2.1
    df.loc[0, "util_docks"] = 0.5

    with pytest.raises(ValueError, match="must be between 0.0 and 2.0"):
        validate_feature_df(df)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("has_neighbors_within_radius", 2),
        ("is_weekend", -1),
        ("is_holiday", 3),
    ],
)
def test_validate_feature_df_rejects_invalid_binary_flags(column, value):
    df = _valid_training_df()
    df.loc[0, column] = value

    with pytest.raises(ValueError, match="must be 0/1"):
        validate_feature_df(df)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("y_stockout_bikes_30", 2),
        ("y_stockout_docks_30", -1),
    ],
)
def test_validate_feature_df_rejects_invalid_training_labels(column, value):
    df = _valid_training_df()
    df.loc[0, column] = value

    with pytest.raises(ValueError, match="must be 0/1"):
        validate_feature_df(df)


@pytest.mark.parametrize(("column", "value"), [("target_bikes_t30", -1), ("target_docks_t30", -2)])
def test_validate_feature_df_rejects_negative_training_targets(column, value):
    df = _valid_training_df()
    df.loc[0, column] = value

    with pytest.raises(ValueError, match="must be non-negative"):
        validate_feature_df(df)


def test_validate_feature_df_allows_null_training_outputs_for_label_maturity():
    df = _valid_training_df()
    df.loc[0, ["y_stockout_bikes_30", "y_stockout_docks_30", "target_bikes_t30", "target_docks_t30"]] = pd.NA

    assert validate_feature_df(df) is True


def test_validate_feature_df_rejects_non_nullable_feature_all_nan():
    df = _valid_training_df()
    df["roll15_net_bikes"] = pd.NA

    with pytest.raises(ValueError, match="entirely null"):
        validate_feature_df(df, missing_rate_threshold=1.0)


def test_validate_feature_df_allows_nullable_feature_columns_to_be_all_nan():
    df = _valid_training_df()
    for column in NULLABLE_FEATURE_COLUMNS:
        df[column] = pd.NA

    assert validate_feature_df(df, missing_rate_threshold=1.0) is True


def test_validate_feature_df_allows_all_weather_columns_to_be_all_nan():
    df = _valid_training_df()
    for column in WEATHER_FEATURE_COLUMNS:
        df[column] = pd.NA

    assert validate_feature_df(df) is True


def test_validate_feature_df_rejects_duplicate_entity_grain():
    df = pd.concat([_valid_training_df(), _valid_training_df()], ignore_index=True)

    with pytest.raises(ValueError, match="entity grain city\\+dt\\+station_id must be unique"):
        validate_feature_df(df)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("hour", 24),
        ("hour", 3.5),
        ("dow", 0),
        ("dow", 7.5),
        ("neighbor_count_within_radius", 2.5),
        ("weather_code", -1),
        ("hourly_weather_code", -2),
    ],
)
def test_validate_feature_df_rejects_invalid_business_ranges(column, value):
    df = _valid_training_df()
    df[column] = df[column].astype("object")
    df.loc[0, column] = value

    with pytest.raises(ValueError, match="must be an integer between"):
        validate_feature_df(df)


def test_validate_feature_df_rejects_missing_rate_above_threshold():
    df = pd.concat([_valid_training_df(), _valid_training_df()], ignore_index=True)
    df.loc[0, "dt"] = "2026-03-11-10-10"
    df.loc[0, ["roll15_net_bikes", "roll30_net_bikes"]] = pd.NA

    with pytest.raises(ValueError, match="feature missing rate"):
        validate_feature_df(
            df,
            missing_rate_threshold=DEFAULT_MISSING_RATE_THRESHOLD,
            per_column_missing_rate_threshold=1.0,
        )


def test_validate_feature_df_rejects_per_column_missing_rate_above_threshold():
    df = pd.concat([_valid_training_df() for _ in range(10)], ignore_index=True)
    for index in range(10):
        df.loc[index, "dt"] = f"2026-03-11-10-{index:02d}"
    for index in range(2):
        df.loc[index, "roll15_net_bikes"] = pd.NA

    with pytest.raises(ValueError, match="feature per-column missing rate exceeded threshold"):
        validate_feature_df(df, per_column_missing_rate_threshold=0.10, missing_rate_threshold=1.0)
