from __future__ import annotations

from typing import Any

import mlflow
import numpy as np
import pandas as pd


class PositiveClassProbabilityModel(mlflow.pyfunc.PythonModel):
    """Serve positive-class probabilities for persisted booster models."""

    def __init__(
        self,
        *,
        model: Any | None = None,
        model_type: str | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self._model = model
        self._model_type = str(model_type or "").lower()
        self._feature_names = list(feature_names or [])

    def _load_external_model(self, model_path: str) -> Any:
        if self._model_type == "xgboost":
            import xgboost as xgb

            model = xgb.Booster()
            model.load_model(model_path)
            return model

        if self._model_type == "lightgbm":
            import lightgbm as lgb

            return lgb.Booster(model_file=model_path)

        raise ValueError(f"unsupported model_type for MLflow pyfunc wrapper: {self._model_type}")

    def _coerce_feature_frame(self, model_input: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input, columns=self._feature_names or None)
        if self._feature_names:
            return model_input[self._feature_names].astype("float64")
        return model_input

    def _predict_xgboost(self, features: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        if isinstance(self._model, xgb.Booster):
            return np.asarray(self._model.predict(xgb.DMatrix(features)), dtype="float64")
        if hasattr(self._model, "predict_proba"):
            return np.asarray(self._model.predict_proba(features)[:, 1], dtype="float64")
        return np.asarray(self._model.predict(features), dtype="float64")

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        config = context.model_config or {}
        if not self._feature_names:
            self._feature_names = list(config.get("feature_names", []))
        if not self._model_type:
            self._model_type = str(config.get("model_type", "")).lower()
        if self._model is not None:
            return
        artifacts = getattr(context, "artifacts", {}) or {}
        model_path = artifacts.get("native_model")
        if not model_path:
            raise ValueError("native_model artifact is required when the model is not embedded in python_model.pkl")
        self._model = self._load_external_model(model_path)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        del context, params

        features = self._coerce_feature_frame(model_input)

        if self._model_type == "xgboost":
            return self._predict_xgboost(features)

        return np.asarray(self._model.predict(features), dtype="float64")


mlflow.models.set_model(PositiveClassProbabilityModel())
