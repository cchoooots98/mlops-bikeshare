import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

import xgboost as xgb

from src.mlflow_pyfunc_model import PositiveClassProbabilityModel


def test_embedded_xgboost_classifier_returns_positive_class_probability():
    features = pd.DataFrame(
        {
            "f1": [0.0, 1.0, 0.0, 1.0],
            "f2": [0.0, 0.0, 1.0, 1.0],
        }
    )
    labels = np.array([0, 0, 1, 1], dtype=int)
    model = xgb.XGBClassifier(
        n_estimators=4,
        max_depth=2,
        learning_rate=0.3,
        subsample=1.0,
        colsample_bytree=1.0,
        tree_method="hist",
        n_jobs=0,
    )
    model.fit(features, labels)

    pyfunc_model = PositiveClassProbabilityModel(
        model=model,
        model_type="xgboost",
        feature_names=["f1", "f2"],
    )

    predictions = pyfunc_model.predict(None, features)

    assert predictions.shape == (4,)
    assert predictions.dtype == np.float64
    assert np.all(predictions >= 0.0)
    assert np.all(predictions <= 1.0)
