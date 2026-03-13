import argparse
import json
import os
from typing import Sequence

import mlflow

from src.model_package import build_model_card_text


def load_eval_summary(eval_json: str | None, run_id: str | None, artifact_path: str) -> dict:
    if bool(eval_json) == bool(run_id):
        raise ValueError("provide exactly one of --eval-json or --run-id")

    if eval_json:
        if not os.path.exists(eval_json):
            raise FileNotFoundError(f"cannot find eval summary file: {eval_json}")
        with open(eval_json, "r", encoding="utf-8") as handle:
            return json.load(handle)

    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    with open(local_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a model card from eval_summary.")
    parser.add_argument("--eval-json", default=None, help="Local eval_summary.json path.")
    parser.add_argument("--run-id", default=None, help="MLflow run_id to download eval_summary from.")
    parser.add_argument("--artifact-path", default="eval/eval_summary.json")
    parser.add_argument("--output", default="model_card.md")
    parser.add_argument("--owner", default="MLOps Team")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--target-side", default="bikes or docks")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> str:
    args = parse_args(argv)
    summary = load_eval_summary(args.eval_json, args.run_id, args.artifact_path)
    if "target_name" not in summary and args.target_side:
        summary = {**summary, "target_name": args.target_side}
    card = build_model_card_text(summary, owner=args.owner, horizon=args.horizon)

    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(card)

    print(f"[OK] Wrote model card to {args.output}")
    return args.output


if __name__ == "__main__":
    main()
