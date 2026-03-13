from __future__ import annotations

from typing import Any

import mlflow
import numpy as np
import pandas as pd


class PositiveClassProbabilityModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        config = context.model_config or {}
        self._feature_names = list(config.get("feature_names", []))
        self._model_type = str(config.get("model_type", "")).lower()
        model_path = context.artifacts["native_model"]

        if self._model_type == "xgboost":
            import xgboost as xgb

            model = xgb.Booster()
            model.load_model(model_path)
            self._model = model
            return

        if self._model_type == "lightgbm":
            import lightgbm as lgb

            self._model = lgb.Booster(model_file=model_path)
            return

        raise ValueError(f"unsupported model_type for MLflow pyfunc wrapper: {self._model_type}")

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        del context, params

        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input, columns=self._feature_names or None)

        features = model_input[self._feature_names].astype("float64") if self._feature_names else model_input

        if self._model_type == "xgboost":
            import xgboost as xgb

            return np.asarray(self._model.predict(xgb.DMatrix(features)), dtype="float64")

        return np.asarray(self._model.predict(features), dtype="float64")


mlflow.models.set_model(PositiveClassProbabilityModel())
