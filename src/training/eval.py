# -*- coding: utf-8 -*-
"""
Post-training evaluation utilities:
1) (Optional) Re-run threshold selection if you saved predictions.
2) Generate model_card.md by auto-filling metrics, context, and assumptions.

Run after training/train.py finishes and logs eval_summary.json.
"""

import argparse
import json
import os
from datetime import datetime, timezone

MODEL_CARD_TEMPLATE = """# Model Card — {model_name}

## Overview
- **Use case**: Predict short-term ({horizon} min) stockout risk for {target_side} at bikeshare stations.
- **Business objective**: Enable proactive rebalancing to reduce customer dissatisfaction and lost trips.
- **Owner**: {owner}
- **Date**: {date_utc}

## Data
- **City**: {city}
- **Time window (UTC)**: {time_start} → {time_end}
- **Sample sizes**: train={n_train}, valid={n_valid}
- **Feature source**: `features_offline` (Athena external table; partitioned by `city`, `dt`)
- **Feature list (first 10)**: {features_head} (total {features_count})
- **Labels**: {label}

## Modeling
- **Algorithm**: {model_type}
- **Primary metric**: PR-AUC (validation) = **{pr_auc_valid:.3f}**
- **Train PR-AUC**: {pr_auc_train:.3f}; **Overfit gap**: {overfit_gap:.3f} (target < 0.10)
- **Threshold (Fβ, β={beta})**: {best_threshold:.2f}
  - Precision={best_precision:.3f}, Recall={best_recall:.3f}, Fβ={best_fbeta:.3f}

## Assumptions & Limitations
- Assumes 5-minute gridded station status and hourly weather aligned to UTC.
- Neighbor features based on spatial BallTree and inverse-distance weighting.
- Labels defined as stockout if `target_*_t30 ≤ threshold` within 30-minute horizon.
- Temporal split ensures validation comes strictly after training period with a gap to reduce leakage.

## Risks
- **Concept drift** (demand shifts, special events, policy changes).
- **Data delays or outages** (GBFS feed, weather source).
- **Imbalanced labels** — monitor precision/recall and threshold regularly.

## Fairness & Ethics
- Model focuses on station-level operational KPIs; does not infer protected attributes about individuals.
- Ensure equitable service levels across neighborhoods; monitor outcomes for systematic bias.

## Monitoring Plan
- Track PR-AUC (rolling window), precision, recall at deployed threshold.
- Alert on metric degradation or data freshness issues.
- Periodic re-training schedule (e.g., weekly) with backtests.

## Reproducibility
- **MLflow experiment**: {experiment}
- **Run ID**: {run_id}
- **Code**: `training/train.py`, `training/eval.py`
- **Data schema**: `schema.py`

"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", default="eval_summary.json", help="Path to eval_summary.json from training run")
    ap.add_argument("--output", default="model_card.md", help="Output model card path")
    ap.add_argument("--owner", default="MLOps Team")
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--target-side", default="bikes (stockout)")
    args = ap.parse_args()

    if not os.path.exists(args.eval_json):
        raise FileNotFoundError(f"Cannot find {args.eval_json}. Run training first.")

    with open(args.eval_json, "r", encoding="utf-8") as f:
        ev = json.load(f)

    # Build the card text
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    features = ev.get("features", [])
    features_head = ", ".join(features[:10])

    card = MODEL_CARD_TEMPLATE.format(
        model_name=f"{ev['model_type']}-{ev['label']}",
        horizon=args.horizon,
        target_side=args.target_side,
        owner=args.owner,
        date_utc=now,
        city=ev["city"],
        time_start=ev["time_start"],
        time_end=ev["time_end"],
        n_train=ev["n_train"],
        n_valid=ev["n_valid"],
        features_head=features_head,
        features_count=len(features),
        label=ev["label"],
        model_type=ev["model_type"],
        pr_auc_valid=ev["pr_auc_valid"],
        pr_auc_train=ev["pr_auc_train"],
        overfit_gap=ev["overfit_gap"],
        beta=ev["beta"],
        best_threshold=ev["best_threshold"],
        best_precision=ev["best_precision"],
        best_recall=ev["best_recall"],
        best_fbeta=ev["best_fbeta"],
        experiment="bikeshare-step4",
        run_id=ev["run_id"],
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(card)

    print(f"[OK] Wrote model card to {args.output}")

if __name__ == "__main__":
    main()
