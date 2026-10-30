"""End-to-End Orchestration Pipeline for Physical Fidelity Auditor (pipeline.py).

Executes the three-tier Physical AI auditing pipeline:
  Tier 1: Ground Truth Generation (NVIDIA Isaac Lab via `npa workbench isaac-lab` or local solver)
  Tier 2: World Model Prediction (NVIDIA Cosmos Predict via `npa workbench cosmos` or local dream emulator)
  Tier 3: Physics Audit (NVIDIA Nemotron on Nebius Token Factory + Tavily Real-World Constants)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from auditor.nemotron_client import (
    NEMOTRON_NANO_DEFAULT,
    TOKEN_FACTORY_DEFAULT_URL,
    NemotronPhysicsAuditor,
)
from auditor.schemas import AuditReport
from auditor.tavily_client import load_dotenv_file

load_dotenv_file()
from ground_truth.scenarios import (
    SCENARIO_REGISTRY,
    SimulationResult,
    generate_ground_truth,
)
from world_model.predictor import (
    HALLUCINATION_PROFILES,
    PredictionResult,
    generate_world_model_prediction,
)

# --- Default Configuration (Nebius Token Factory & Tavily) ---
TOKEN_FACTORY_URL = os.getenv("TOKEN_FACTORY_URL", TOKEN_FACTORY_DEFAULT_URL)
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
DEFAULT_MODEL = os.getenv("NEMOTRON_NANO_MODEL", NEMOTRON_NANO_DEFAULT)


def run_scenario_pipeline(
    scenario: str = "ramp_roll",
    profile: str = "cosmos_default",
    data_dir: Path | str = Path("data"),
    friction: Optional[float] = None,
    mass: Optional[float] = None,
    initial_velocity: Optional[float] = None,
    num_frames: int = 101,
    use_nebius: bool = False,
    nebius_api_key: Optional[str] = None,
    tavily_api_key: Optional[str] = None,
    escalate_ambiguous: bool = True,
) -> Dict[str, Any]:
    """Execute the complete 5-step physics audit pipeline for a single scenario."""
    t_start = time.perf_counter()
    base_dir = Path(data_dir)
    gt_root = base_dir / "ground_truth"
    pred_root = base_dir / "predictions"
    reports_root = base_dir / "reports"

    # --- Step 1: Generate Ground Truth (Isaac Lab) ---
    gt_result: SimulationResult = generate_ground_truth(
        scenario=scenario,
        output_root=gt_root,
        friction=friction,
        mass=mass,
        initial_velocity=initial_velocity,
        num_frames=num_frames,
        use_nebius=use_nebius,
    )

    # --- Step 2: Generate World Model Predictions (Cosmos Predict) ---
    pred_result: PredictionResult = generate_world_model_prediction(
        scenario=scenario,
        ground_truth_root=gt_root,
        output_root=pred_root,
        profile=profile,
        num_frames=num_frames,
        use_nebius=use_nebius,
    )

    # --- Steps 3, 4 & 5: Retrieve Constants (Tavily), Audit (Nemotron), & Save Report ---
    auditor = NemotronPhysicsAuditor(
        api_key=NEBIUS_API_KEY if nebius_api_key is None else nebius_api_key,
        base_url=TOKEN_FACTORY_URL,
        nano_model=DEFAULT_MODEL,
        tavily_api_key=TAVILY_API_KEY if tavily_api_key is None else tavily_api_key,
    )
    audit_report: AuditReport = auditor.audit_scenario(
        scenario=scenario,
        ground_truth_root=gt_root,
        predictions_root=pred_root,
        output_root=reports_root,
        escalate_ambiguous=escalate_ambiguous,
    )

    total_elapsed = round(time.perf_counter() - t_start, 3)
    return {
        "scenario": scenario,
        "profile": profile,
        "ground_truth": gt_result,
        "prediction": pred_result,
        "audit_report": audit_report,
        "pipeline_elapsed_seconds": total_elapsed,
    }


def run_full_benchmark(
    profile: str = "cosmos_default",
    data_dir: Path | str = Path("data"),
    num_frames: int = 101,
    use_nebius: bool = False,
    nebius_api_key: Optional[str] = None,
    tavily_api_key: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run all three benchmark scenarios (ramp_roll, block_stack, projectile) and aggregate scores."""
    base_dir = Path(data_dir)
    results: Dict[str, Dict[str, Any]] = {}
    for scenario_name in SCENARIO_REGISTRY:
        results[scenario_name] = run_scenario_pipeline(
            scenario=scenario_name,
            profile=profile,
            data_dir=base_dir,
            num_frames=num_frames,
            use_nebius=use_nebius,
            nebius_api_key=nebius_api_key,
            tavily_api_key=tavily_api_key,
        )

    # Write aggregate summary JSON
    reports_root = base_dir / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    reports_map = {k: v["audit_report"].model_dump() for k, v in results.items()}
    mean_phr = round(
        sum(r["physics_hallucination_rate"] for r in reports_map.values()) / max(1, len(reports_map)),
        4,
    )
    mean_pir = round(
        sum(r["physical_integrity_rating"] for r in reports_map.values()) / max(1, len(reports_map)),
        4,
    )
    total_debt = round(
        sum(r["kinetic_debt_joules"] for r in reports_map.values()),
        3,
    )

    aggregate = {
        "project": "physics-audit",
        "world_model_profile": profile,
        "scenarios_audited": len(reports_map),
        "mean_physics_hallucination_rate": mean_phr,
        "mean_physical_integrity_rating": mean_pir,
        "total_kinetic_debt_joules": total_debt,
        "target_phr_threshold": 0.10,
        "target_pir_threshold": 0.85,
        "reports": reports_map,
    }
    (reports_root / "aggregate_audit_summary.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Physical Fidelity Auditor (physics-audit) End-to-End Pipeline."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["all", *SCENARIO_REGISTRY.keys()],
        help="Scenario to run ('ramp_roll', 'block_stack', 'projectile', or 'all').",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="cosmos_default",
        choices=list(HALLUCINATION_PROFILES.keys()),
        help="World model behavior profile.",
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Root directory for artifacts and reports.")
    parser.add_argument("--num-frames", type=int, default=101, help="Number of frames per scenario.")
    parser.add_argument("--friction", type=float, default=None, help="Optional friction coefficient override.")
    parser.add_argument("--mass", type=float, default=None, help="Optional mass override (kg).")
    parser.add_argument("--initial-velocity", type=float, default=None, help="Optional initial velocity override (m/s).")
    parser.add_argument(
        "--use-nebius",
        action="store_true",
        help="Invoke Nebius `npa workbench` CLI for Isaac Lab and Cosmos jobs when installed.",
    )
    args = parser.parse_args()

    if args.scenario == "all":
        results = run_full_benchmark(
            profile=args.profile,
            data_dir=args.data_dir,
            num_frames=args.num_frames,
            use_nebius=args.use_nebius,
        )
        print("\n=== PHYSICAL FIDELITY AUDITOR BENCHMARK SUMMARY ===")
        for name, bundle in results.items():
            rep: AuditReport = bundle["audit_report"]
            print(
                f"  [{name:11s}] Verdict: {rep.overall_verdict:4s} | "
                f"PHR: {rep.physics_hallucination_rate:.3f} | "
                f"PIR: {rep.physical_integrity_rating:.3f} | "
                f"Kinetic Debt: {rep.kinetic_debt_joules:6.1f}J | "
                f"Violations: {rep.violated_frames_count}/{rep.total_frames} frames "
                f"({bundle['pipeline_elapsed_seconds']:.2f}s)"
            )
    else:
        bundle = run_scenario_pipeline(
            scenario=args.scenario,
            profile=args.profile,
            data_dir=args.data_dir,
            friction=args.friction,
            mass=args.mass,
            initial_velocity=args.initial_velocity,
            num_frames=args.num_frames,
            use_nebius=args.use_nebius,
        )
        rep = bundle["audit_report"]
        print(
            f"\n[PhysicsAudit] Scenario={rep.scenario} | Verdict={rep.overall_verdict} | "
            f"PHR={rep.physics_hallucination_rate:.3f} | PIR={rep.physical_integrity_rating:.3f} | "
            f"Violations={rep.violated_frames_count}/{rep.total_frames}"
        )


if __name__ == "__main__":
    main()

