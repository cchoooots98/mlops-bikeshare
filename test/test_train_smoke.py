import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("mlflow")

from pipelines.register_model import register_model_version  # noqa: E402
from src.features.schema import TRAINING_REQUIRED_COLUMNS  # noqa: E402
from src.model_package import load_package_manifest  # noqa: E402
from src.model_target import target_spec_from_predict_bikes  # noqa: E402
from src.training import eval as train_eval  # noqa: E402
from src.training import train  # noqa: E402


def _make_training_df() -> pd.DataFrame:
    rows = []
    for index in range(24):
        bikes = 5 + (index % 6)
        row = {
            "city": "paris",
            "dt": f"2026-03-01-{(index // 12):02d}-{(index % 12) * 5:02d}",
            "station_id": f"station-{index % 4:03d}",
            "capacity": 40,
            "lat": 48.8566,
            "lon": 2.3522,
            "bikes": bikes,
            "docks": 40 - bikes,
            "minutes_since_prev_snapshot": 5.0,
            "util_bikes": bikes / 40.0,
            "util_docks": (40 - bikes) / 40.0,
            "delta_bikes_5m": float((index % 3) - 1),
            "delta_docks_5m": float(1 - (index % 3)),
            "roll15_net_bikes": float(index % 4),
            "roll30_net_bikes": float(index % 5),
            "roll60_net_bikes": float(index % 6),
            "roll15_bikes_mean": float(bikes),
            "roll30_bikes_mean": float(bikes + 1),
            "roll60_bikes_mean": float(bikes + 2),
            "nbr_bikes_weighted": float(bikes + 0.5),
            "nbr_docks_weighted": float(39 - bikes),
            "has_neighbors_within_radius": 1,
            "neighbor_count_within_radius": 4,
            "hour": index % 24,
            "dow": 2,
            "is_weekend": 0,
            "is_holiday": 0,
            "temperature_c": 15.0 + (index % 3),
            "humidity_pct": 55.0,
            "wind_speed_ms": 4.0,
            "precipitation_mm": 0.1,
            "weather_code": 3,
            "hourly_temperature_c": 15.5,
            "hourly_humidity_pct": 54.0,
            "hourly_wind_speed_ms": 4.2,
            "hourly_precipitation_mm": 0.2,
            "hourly_precipitation_probability_pct": 30.0,
            "hourly_weather_code": 3,
            "y_stockout_bikes_30": 1 if index % 2 == 0 else 0,
            "y_stockout_docks_30": 1 if index % 3 == 0 else 0,
            "target_bikes_t30": max(0, bikes - 2),
            "target_docks_t30": min(40, 35 - bikes),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=TRAINING_REQUIRED_COLUMNS)


def _configure_training_sources(monkeypatch, base_df: pd.DataFrame, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    train.mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(train, "create_pg_engine", lambda pg_config: object())
    monkeypatch.setattr(
        train,
        "list_unique_dt_postgres",
        lambda engine, pg_config, city, start_dt, end_dt: sorted(base_df["dt"].unique()),
    )

    def fake_load_slice(engine, data_config, select_columns, start_dt, end_dt):
        mask = (base_df["dt"] >= start_dt) & (base_df["dt"] <= end_dt)
        return base_df.loc[mask, select_columns].copy()

    monkeypatch.setattr(train, "load_slice_postgres", fake_load_slice)


def test_build_feature_select_query_reads_postgres_feature_table_in_schema_order():
    columns = ["capacity", "lat", "lon", "bikes", "docks", "minutes_since_prev_snapshot", "y_stockout_bikes_30"]
    sql = train.build_feature_select_query("analytics", "feat_station_snapshot_5min", columns)

    assert 'FROM "analytics"."feat_station_snapshot_5min"' in sql
    assert '"capacity" AS "capacity", "lat" AS "lat", "lon" AS "lon"' in sql
    assert '"y_stockout_bikes_30" AS "y_stockout_bikes_30"' in sql
    assert "WHERE city = :city AND dt >= :start_dt AND dt <= :end_dt" in sql


def test_compute_temporal_split_applies_gap_minutes_on_5min_grid():
    dt_list = [f"2026-03-01-00-{minute:02d}" for minute in range(0, 50, 5)]
    split = train.compute_temporal_split(dt_list, valid_ratio=0.2, gap_minutes=15)

    assert split.train_end_dt == "2026-03-01-00-35"
    assert split.valid_start_dt == "2026-03-01-00-45"
    assert split.gap_ticks == 3


@pytest.mark.parametrize("predict_bikes", [True, False])
def test_train_smoke_logs_model_and_registers_artifact(monkeypatch, tmp_path, predict_bikes):
    base_df = _make_training_df()
    target_spec = target_spec_from_predict_bikes(predict_bikes)
    _configure_training_sources(monkeypatch, base_df, tmp_path)

    data_config = train.DataConfig(
        city="paris",
        start="2026-03-01 00:00",
        end="2026-03-01 01:55",
        pg_host="localhost",
        pg_port=15432,
        pg_db="velib_dw",
        pg_user="velib",
        pg_password="velib",
    )
    train_config = train.TrainConfig(
        predict_bikes=predict_bikes,
        model_type="xgboost",
        experiment="pytest-train-smoke",
        gap_minutes=0,
        xgb_params={
            "n_estimators": 8,
            "max_depth": 3,
            "learning_rate": 0.2,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "tree_method": "hist",
            "n_jobs": 0,
        },
    )

    result = train.run_training_pipeline(data_config, train_config)
    registration = register_model_version(
        run_id=result["run_id"],
        model_name=result["model_name"],
        artifact_path=result["model_artifact_path"],
    )

    assert result["model_artifact_path"] == "model"
    assert result["run_id"]
    assert result["predict_bikes"] is predict_bikes
    assert result["label"] == target_spec.label_column
    assert result["score_column"] == target_spec.score_column
    assert result["package_dir"]
    assert result["package_manifest_path"]
    manifest = load_package_manifest(result["package_dir"])
    assert manifest["best_threshold"] == result["best_threshold"]
    assert manifest["feature_columns"] == train.FEATURE_COLUMNS
    model_card = (Path(result["package_dir"]) / "artifacts" / "model_card.md").read_text(encoding="utf-8")
    assert "radius-based neighbor graph" in model_card
    assert "Serving artifact path" in model_card
    mlmodel_text = (Path(result["package_dir"]) / "model" / "MLmodel").read_text(encoding="utf-8")
    assert "loader_module: mlflow.pyfunc.model" in mlmodel_text
    assert "model_code_path:" not in mlmodel_text
    assert (Path(result["package_dir"]) / "model" / "python_model.pkl").exists()
    assert not (Path(result["package_dir"]) / "model" / "mlflow_prob_model.py").exists()
    assert b"src.mlflow_pyfunc_model" in (Path(result["package_dir"]) / "model" / "python_model.pkl").read_bytes()
    assert b"src.training" not in (Path(result["package_dir"]) / "model" / "python_model.pkl").read_bytes()
    assert int(registration["version"]) >= 1


@pytest.mark.parametrize("predict_bikes", [True, False])
def test_training_drops_rows_with_null_selected_label_or_paired_target(monkeypatch, tmp_path, predict_bikes):
    base_df = _make_training_df()
    target_spec = target_spec_from_predict_bikes(predict_bikes)
    base_df.loc[3, target_spec.label_column] = None
    base_df.loc[5, target_spec.paired_target_column] = None
    base_df.loc[20, target_spec.label_column] = None
    base_df.loc[22, target_spec.paired_target_column] = None
    _configure_training_sources(monkeypatch, base_df, tmp_path)

    data_config = train.DataConfig(
        city="paris",
        start="2026-03-01 00:00",
        end="2026-03-01 01:55",
        pg_host="localhost",
        pg_port=15432,
        pg_db="velib_dw",
        pg_user="velib",
        pg_password="velib",
    )
    train_config = train.TrainConfig(
        predict_bikes=predict_bikes,
        model_type="xgboost",
        valid_ratio=0.25,
        gap_minutes=0,
        experiment="pytest-train-null-output-filter",
        xgb_params={
            "n_estimators": 8,
            "max_depth": 3,
            "learning_rate": 0.2,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "tree_method": "hist",
            "n_jobs": 0,
        },
    )

    result = train.run_training_pipeline(data_config, train_config)

    assert result["paired_target"] == target_spec.paired_target_column
    assert result["n_train"] == 16
    assert result["n_valid"] == 4
    assert result["dropped_train_rows_null_outputs"] == 2
    assert result["dropped_valid_rows_null_outputs"] == 2


@pytest.mark.parametrize("predict_bikes", [True, False])
def test_training_rejects_single_class_validation_slice(monkeypatch, tmp_path, predict_bikes):
    base_df = _make_training_df()
    target_spec = target_spec_from_predict_bikes(predict_bikes)
    base_df.loc[base_df.index >= 18, target_spec.label_column] = 0
    _configure_training_sources(monkeypatch, base_df, tmp_path)

    with pytest.raises(RuntimeError, match="validation slice .* must contain both classes"):
        train.run_training_pipeline(
            train.DataConfig(
                city="paris",
                start="2026-03-01 00:00",
                end="2026-03-01 01:55",
                pg_host="localhost",
                pg_port=15432,
                pg_db="velib_dw",
                pg_user="velib",
                pg_password="velib",
            ),
            train.TrainConfig(
                predict_bikes=predict_bikes,
                model_type="xgboost",
                experiment="pytest-train-single-class",
                gap_minutes=0,
                xgb_params={
                    "n_estimators": 8,
                    "max_depth": 3,
                    "learning_rate": 0.2,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "tree_method": "hist",
                    "n_jobs": 0,
                },
            ),
        )


def test_eval_writes_model_card_from_minimal_eval_summary(tmp_path):
    eval_summary_path = tmp_path / "eval_summary.json"
    eval_summary_path.write_text(
        json.dumps(
            {
                "run_id": "run-123",
                "experiment": "pytest-exp",
                "model_name": "paris_y_stockout_bikes_30_xgboost",
                "model_type": "xgboost",
                "label": "y_stockout_bikes_30",
                "city": "paris",
                "time_start": "2026-03-01 00:00",
                "time_end": "2026-03-07 23:55",
                "feature_source": "analytics.feat_station_snapshot_5min",
                "feature_contract": "v1_dim_weather_aligned",
                "features": ["minutes_since_prev_snapshot", "util_bikes"],
                "n_train": 100,
                "n_valid": 20,
                "pr_auc_train": 0.82,
                "pr_auc_valid": 0.76,
                "overfit_gap": 0.06,
                "best_threshold": 0.33,
                "best_precision": 0.5,
                "best_recall": 0.7,
                "best_fbeta": 0.65,
                "beta": 2.0,
                "model_artifact_path": "model",
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "model_card.md"

    train_eval.main(["--eval-json", str(eval_summary_path), "--output", str(output_path)])
    content = output_path.read_text(encoding="utf-8")

    assert "analytics.feat_station_snapshot_5min" in content
    assert "radius-based neighbor graph" in content
    assert "Serving artifact path" in content
