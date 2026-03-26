import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from src.model_package import load_deployment_state, write_deployment_state


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rollback an active deployment-state file to a previous known-good state."
    )
    parser.add_argument("--target-name", required=True, choices=["bikes", "docks"])
    parser.add_argument("--environment", required=True)
    parser.add_argument("--from-state", required=True, help="Active deployment state path to overwrite.")
    parser.add_argument("--to-state", required=True, help="Rollback source deployment state path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    active_state_path = Path(args.from_state)
    rollback_state_path = Path(args.to_state)

    current_state = load_deployment_state(active_state_path)
    rollback_state = load_deployment_state(rollback_state_path)

    restored_state = {
        **rollback_state,
        "environment": args.environment,
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "rollback",
        "target_name": args.target_name,
    }
    output_path = write_deployment_state(active_state_path, restored_state)
    result = {
        "target_name": args.target_name,
        "environment": args.environment,
        "active_state_path": str(active_state_path.resolve()),
        "rollback_source_path": str(rollback_state_path.resolve()),
        "restored_deployment_state_path": output_path,
        "previous_run_id": current_state.get("run_id"),
        "restored_run_id": restored_state.get("run_id"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
