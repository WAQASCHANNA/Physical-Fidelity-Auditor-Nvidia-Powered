"""Path A & B Validation Script: npa/examples/isaac_franka_token_factory_reason.py.

Calls `npa.workbench.token_factory.reason_scene` to produce `scene_reasoning.json`
and `metadata.json` using `nvidia/Cosmos3-Super-Reasoner` on Nebius Token Factory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npa.workbench.token_factory import reason_scene


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Nebius Token Factory Cosmos3-Super-Reasoner on Isaac Lab Franka Lift-Cube frames."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="docs/assets/hackathon/isaac-franka-lift-cube",
        help="Input directory or S3 path containing ground-truth PNG frames.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/tmp/isaac-franka-reason-out",
        help="Output directory or S3 path for scene_reasoning.json and metadata.json.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Cosmos3-Super-Reasoner",
        help="Token Factory model name.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="cosmos_default",
        help="World model hallucination profile ('cosmos_default' or 'physically_solvent').",
    )
    args = parser.parse_args()

    result = reason_scene(
        input_path=args.input_path,
        output_path=args.output_path,
        model=args.model,
        hallucination_profile=args.profile,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
