# All comments in English

import mlflow
import pandas as pd

from src.features.schema import FEATURE_COLUMNS

model = mlflow.pyfunc.load_model(
    "e:/算法自学/End2EndProject/mlops-bikeshare/mlruns/1/models/m-02fa82813dbe4fbcab848468b9d1e744/artifacts"
)
# Build a tiny test frame in the EXACT feature order
X = pd.DataFrame({c: [0.0] for c in FEATURE_COLUMNS}, dtype="float64")[FEATURE_COLUMNS]

# Should return a 1-D float array (probabilities)
y = model.predict(X)
print(y)
