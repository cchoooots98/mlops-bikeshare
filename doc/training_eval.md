# Training & Evaluation Strategy

## 1. Data Splitting
- **Temporal split (time extrapolation):**
  - Training: N days.
  - Validation: subsequent days/weeks.
  - Ensures no leakage from the future into the past.

- **Example:** Train on 2025-08-18 to 2025-08-21, validate on 2025-08-22.

---

## 2. Imbalanced Data Handling
- **Problem:** Stockout events (positive class) are rare.
- **Mitigations:**
  - Adjust decision threshold to maximize PR-AUC or Fβ (β=2).
  - Use `class_weight=balanced` (XGBoost/LightGBM).
  - Track **PR-AUC** and minority-class F1 as primary metrics.

---

## 3. Models and Logging
- Models: XGBoost, LightGBM.
- Auto-logging with MLflow / SageMaker Experiments.
- Artifacts saved: metrics, confusion matrix, feature importances, model object.

---

## 4. Evaluation Metrics
- **Primary:**
  - PR-AUC ≥ 0.70
  - F1 (stockout bikes) ≥ 0.55
- **Secondary:**
  - AUC-ROC
  - Calibration curves
  - Regression RMSE for `target_bikes_t30` and `target_docks_t30`.

---

## 5. Feature Importance and Stability
- Compute feature importance (gain/weight).
- Verify stability across splits.
- Check for feature leakage (esp. time-aligned weather).

---

## 6. Consistency Between Offline and Online
- Offline → online features aligned at 5-minute intervals.
- Feature schema enforced by `schema.py` (`validate_feature_df`).
- Allowed drift thresholds monitored by Model Monitor (PSI < 0.2).

---

## 7. Experiment Tracking
- Each run tagged with:
  - City (`nyc`, `paris`, …)
  - Date range
  - Feature version
  - Model parameters
- Model card auto-filled with metrics, assumptions, risks.

---
