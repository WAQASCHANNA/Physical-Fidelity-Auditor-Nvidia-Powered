"""CLI entrypoint for `npa workbench token-factory reason` (npa/cli.py).

Supports:
  python -m npa.cli workbench token-factory reason \
    --input-path s3://physics-audit/ground_truth/franka_lift/frames/ \
    --output-path s3://physics-audit/predictions/franka_lift/ \
    --model nvidia/Cosmos3-Super-Reasoner \
    --output json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npa.workbench.token_factory import reason_scene


def main() -> None:
    parser = argparse.ArgumentParser(prog="npa", description="Nebius Physical AI Workbench CLI")
    sub = parser.add_subparsers(dest="command")

    wb = sub.add_parser("workbench", help="Physical AI Workbench commands")
    wb_sub = wb.add_subparsers(dest="tool")

    tf = wb_sub.add_parser("token-factory", help="Nebius Token Factory hosted inference")
    tf_sub = tf.add_subparsers(dest="action")

    reason_p = tf_sub.add_parser("reason", help="Run Cosmos3-Super-Reasoner scene understanding & planning")
    reason_p.add_argument("--input-path", required=True, help="Input directory or S3 URI of ground-truth frames")
    reason_p.add_argument("--output-path", required=True, help="Output directory or S3 URI for scene_reasoning.json")
    reason_p.add_argument("--model", default="nvidia/Cosmos3-Super-Reasoner", help="Token Factory model ID")
    reason_p.add_argument("--profile", default="cosmos_default", help="World model profile ('cosmos_default' or 'physically_solvent')")
    reason_p.add_argument("--output", default="json", choices=["json", "text"], help="Output format")

    args = parser.parse_args()
    if args.command == "workbench" and args.tool == "token-factory" and args.action == "reason":
        result = reason_scene(
            input_path=args.input_path,
            output_path=args.output_path,
            model=args.model,
            hallucination_profile=args.profile,
        )
        if args.output == "json":
            print(json.dumps(result, indent=2))
        else:
            print(f"[npa token-factory reason] status={result['status']} -> {args.output_path}/scene_reasoning.json")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

