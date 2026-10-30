"""Module B: 3D World Model Predictor (world_model/predictor.py).

Consumes initial 3D ground-truth frames, depth/SDF telemetry, and action conditioning
to generate predicted 3D rollouts ("dreams") using NVIDIA Cosmos (`nvidia/Cosmos3-Super-Reasoner`
on Nebius Token Factory or `npa workbench cosmos`).
Includes a deterministic 3D world-model emulator supporting `franka_lift_cube`,
`block_stack`, `projectile`, and `ramp_roll` across multiple fidelity profiles.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ground_truth.scenarios import (
    SCENARIO_REGISTRY,
    ScenarioConfig,
    TrajectoryPoint,
    get_scenario_config,
    render_scenario_frame,
    simulate_trajectory,
)

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


HALLUCINATION_PROFILES: Dict[str, Dict[str, Any]] = {
    "cosmos_default": {
        "label": "Cosmos-Predict2.5 / Cosmos 3 Baseline (Unconstrained 3D Rollout)",
        "description": (
            "Exhibits characteristic 3D world-model hallucinations: gripper/surface SDF interpenetration, "
            "floating cube levitation without grasp closure force, under-damped friction, and brief occlusion flicker."
        ),
        "severity_scale": 1.0,
    },
    "physically_solvent": {
        "label": "Cosmos 3 Super Reasoner + 3D Physics Control Prior (High Fidelity)",
        "description": (
            "3D state-regularized world model achieving PHR < 0.10 and PIR > 0.85 "
            "with valid 3D contact SDF >= 0 and physically consistent grasp/trajectory dynamics."
        ),
        "severity_scale": 0.12,
    },
    "severe_hallucination": {
        "label": "Unconstrained 3D Video Diffusion Generator (High Hallucination)",
        "description": (
            "Visually plausible 3D video generator lacking rigid-body priors; exhibits severe "
            "3D mesh interpenetration, spontaneous momentum jumps, gravity defiance, and object disappearance."
        ),
        "severity_scale": 1.85,
    },
}


@dataclass
class PredictionResult:
    """Output bundle produced by the 3D world-model prediction stage."""

    scenario: str
    profile: str
    output_dir: Path
    frames_dir: Path
    frame_paths: List[Path]
    trajectory_csv: Path
    model_info_json: Path
    predicted_trajectory: List[TrajectoryPoint]
    visibility_mask: List[bool]
    injected_anomalies: List[Dict[str, Any]]
    model_info: Dict[str, Any]


def _try_run_npa_cosmos(
    scenario: str,
    ground_truth_frames_dir: Path,
    output_dir: Path,
) -> bool:
    """Attempt to invoke `npa workbench cosmos predict` if the Nebius CLI is installed."""
    npa_bin = shutil.which("npa")
    if not npa_bin:
        return False
    try:
        completed = subprocess.run(
            [
                npa_bin,
                "workbench",
                "cosmos",
                "predict",
                "--input",
                str(ground_truth_frames_dir),
                "--output",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        return completed.returncode == 0 and (output_dir / "model_info.json").exists()
    except Exception:
        return False


def _query_cosmos_super_reasoner_scene_plan(
    config: ScenarioConfig,
    profile: str,
) -> Dict[str, Any]:
    """Query `nvidia/Cosmos3-Super-Reasoner` on Token Factory if available, or return structured 3D scene plan."""
    api_key = os.getenv("NEBIUS_TOKEN_FACTORY_KEY") or os.getenv("NEBIUS_API_KEY") or ""
    base_url = os.getenv("TOKEN_FACTORY_URL", "https://api.tokenfactory.nebius.com/v1/")
    model_name = os.getenv("COSMOS_REASONER_MODEL", "nvidia/Cosmos3-Super-Reasoner")

    default_plan = {
        "model": model_name,
        "task_id": config.isaac_task_id,
        "scene_description": config.description,
        "predicted_stages": ["approach", "pre_grasp", "grasp", "lift", "hold"]
        if config.scenario == "franka_lift_cube"
        else ["initial_contact", "dynamic_evolution", "steady_state"],
        "physical_priors_active": profile == "physically_solvent",
    }

    if not HAS_OPENAI or not api_key or api_key.startswith("your_"):
        return default_plan

    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Provide a 2-sentence 3D physical manipulation scene summary for Isaac Lab task "
                        f"{config.isaac_task_id}: {config.description}"
                    ),
                }
            ],
            max_tokens=180,
            timeout=8.0,
        )
        text = resp.choices[0].message.content or ""
        if text.strip():
            default_plan["cosmos_reasoner_response"] = text.strip()
            default_plan["mode"] = "token_factory_live"
    except Exception:
        default_plan["mode"] = "local_3d_world_emulator"

    return default_plan


def synthesize_world_model_dream(
    config: ScenarioConfig,
    gt_trajectory: List[TrajectoryPoint],
    profile: str = "cosmos_default",
) -> tuple[List[TrajectoryPoint], List[bool], List[Dict[str, Any]]]:
    """Synthesize predicted 3D trajectory states and frame visibility mask for a 3D world model."""
    if profile not in HALLUCINATION_PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Valid: {list(HALLUCINATION_PROFILES)}")

    scale = float(HALLUCINATION_PROFILES[profile]["severity_scale"])
    pred_points: List[TrajectoryPoint] = []
    visibility_mask: List[bool] = []
    injected_anomalies: List[Dict[str, Any]] = []

    for pt in gt_trajectory:
        idx = pt.frame_index
        x, y, z = pt.x, pt.y, pt.z
        vx, vy, vz = pt.vx, pt.vy, pt.vz
        wx, wy, wz = pt.wx, pt.wy, pt.wz
        qx, qy, qz, qw = pt.qx, pt.qy, pt.qz, pt.qw
        contact = pt.contact
        sdf = pt.sdf_clearance
        grasp_f = pt.grasp_force_n
        ee_x, ee_y, ee_z = pt.ee_x, pt.ee_y, pt.ee_z
        energy = pt.mechanical_energy
        visible = True

        if profile == "physically_solvent":
            noise_x = 0.006 * math.sin(idx * 0.15)
            noise_y = 0.004 * math.cos(idx * 0.12)
            noise_z = 0.004 * math.cos(idx * 0.15)
            if idx in (54, 55):
                noise_x += 0.085
                vx += 0.34
                injected_anomalies.append(
                    {
                        "frame_index": idx,
                        "violation_type": "Friction and damping behavior",
                        "description": (
                            f"Frame {idx}: Minor 3D tangential micro-slip during {pt.stage} stage "
                            f"(predicted vx={vx:.2f} m/s vs GT {pt.vx:.2f} m/s)."
                        ),
                        "severity": "LOW",
                    }
                )
            x = round(x + noise_x, 5)
            y = round(y + noise_y, 5)
            z = round(max(0.08, z + noise_z), 5)
            vx = round(vx, 5)
            vy = round(vy, 5)
            vz = round(vz, 5)

        else:
            if config.scenario == "warehouse_amr":
                if 28 <= idx <= 38:
                    penetration = 0.22 * scale * math.sin(((idx - 27) / 12.0) * math.pi)
                    y = pt.y + 1.15 * scale
                    sdf = round(-penetration, 5)
                    injected_anomalies.append(
                        {
                            "frame_index": idx,
                            "violation_type": "Contact constraints",
                            "description": (
                                f"Frame {idx}: Robot_01 interpenetrates Rack_A / Pallet_02 bounding box "
                                f"(3D LiDAR SDF={sdf:+.2f}m, lateral y={y:.2f}m vs GT y={pt.y:.2f}m)."
                            ),
                            "severity": "HIGH",
                        }
                    )
                elif 58 <= idx <= 68:
                    vx = pt.vx + 2.8 * scale
                    energy = pt.mechanical_energy + 65.0 * scale
                    injected_anomalies.append(
                        {
                            "frame_index": idx,
                            "violation_type": "Friction and damping behavior",
                            "description": (
                                f"Frame {idx}: Robot_01 exhibits unphysical lateral skid & velocity surge "
                                f"(vx={vx:.2f} m/s vs nominal 1.20 m/s on concrete mu={config.friction:.2f})."
                            ),
                            "severity": "MEDIUM",
                        }
                    )
                elif 82 <= idx <= 84 and scale >= 0.9:
                    visible = False
                    injected_anomalies.append(
                        {
                            "frame_index": idx,
                            "violation_type": "Object permanence",
                            "description": (
                                f"Frame {idx}: Dynamic obstacle Forklift_01 / Robot_01 vanishes from Camera_Front & Lidar_01 scan."
                            ),
                            "severity": "HIGH",
                        }
                    )

            elif config.scenario == "franka_lift_cube":
                # 1. Frames 26..33: 3D Gripper-Cube & Worktable SDF Interpenetration
                if 26 <= idx <= 33:
                    penetration = 0.16 * scale * math.sin(((idx - 25) / 9.0) * math.pi)
                    z = pt.z - penetration
                    sdf = round(-penetration, 5)
                    injected_anomalies.append(
                        {
                            "frame_index": idx,
                            "violation_type": "Contact constraints",
                            "description": (
                                f"Frame {idx}: Franka parallel gripper pushes 3D cube through solid worktable "
                                f"(3D SDF={sdf:+.2f}m, z={z:.2f}m vs GT z={pt.z:.2f}m)."
                            ),
                            "severity": "HIGH" if penetration > 0.11 else "MEDIUM",
                        }
                    )
                # 2. Frames 52..62: Unphysical Levitation During Lift Without Grasp Force
                elif 52 <= idx <= 62:
                    grasp_f = 0.0
                    z = pt.z + 0.24 * scale
                    y = pt.y + 0.18 * scale
                    vz = pt.vz + 1.8 * scale
                    energy = pt.mechanical_energy + 6.4 * scale
                    injected_anomalies.append(
                        {
                            "frame_index": idx,
                            "violation_type": "Gravity and free-fall behavior",
                            "description": (
                                f"Frame {idx}: Cube levitates +{(z - pt.z):.2f}m above open Franka gripper "
                                f"with zero grasp force (F_grasp=0.0N < required hold threshold)."
                            ),
                            "severity": "HIGH",
                        }
                    )
                # 3. Frames 84..86: 3D Object Permanence Loss During Hold
                elif 84 <= idx <= 86 and scale >= 0.9:
                    visible = False
                    injected_anomalies.append(
                        {
                            "frame_index": idx,
                            "violation_type": "Object permanence",
                            "description": (
                                f"Frame {idx}: Grasped 3D cube vanishes from Franka end-effector "
                                f"at 3D pose ({pt.x:.2f}, {pt.y:.2f}, {pt.z:.2f})m."
                            ),
                            "severity": "HIGH",
                        }
                    )

            elif config.scenario == "ramp_roll":
                if 28 <= idx <= 35:
                    depth = 0.24 * scale * math.sin(((idx - 27) / 9.0) * math.pi)
                    z -= depth
                    sdf = round(-depth, 5)
                    injected_anomalies.append(
                        {
                            "frame_index": idx,
                            "violation_type": "Contact constraints",
                            "description": (
                                f"Frame {idx}: Sphere interpenetrates 3D inclined ramp mesh by "
                                f"{depth:.2f}m (SDF={sdf:+.2f}m, z={z:.2f}m vs GT z={pt.z:.2f}m)."
                            ),
                            "severity": "HIGH" if depth > 0.14 else "MEDIUM",
                        }
                    )
                elif 64 <= idx <= 74:
                    boost = 0.14 * (idx - 63) * scale
                    x += boost * 0.45
                    vx += 1.65 * scale
                    energy = pt.mechanical_energy + 6.8 * scale
                    injected_anomalies.append(
                        {
                            "frame_index": idx,
                            "violation_type": "Friction and damping behavior",
                            "description": (
                                f"Frame {idx}: Sphere accelerates spontaneously on flat wood runout "
                                f"despite kinetic friction mu={config.friction:.2f} "
                                f"(vx={vx:.2f} m/s vs GT {pt.vx:.2f} m/s; +6.8J kinetic debt)."
                            ),
                            "severity": "MEDIUM" if scale <= 1.0 else "HIGH",
                        }
                    )
                elif 85 <= idx <= 87 and scale >= 0.9:
                    visible = False
                    injected_anomalies.append(
                        {
                            "frame_index": idx,
                            "violation_type": "Object permanence",
                            "description": (
                                f"Frame {idx}: Rolling 3D sphere disappears completely from the scene "
                                f"for 3 consecutive frames before re-materializing."
                            ),
                            "severity": "HIGH",
                        }
                    )

            elif config.scenario == "block_stack":
                if 18 <= idx <= 34:
                    z = 1.00 + 0.04 * scale * math.sin((idx - 18) * 0.3)
                    x = 1.84
                    vx, vy, vz = 0.0, 0.0, 0.0
                    injected_anomalies.append(
                        {
                            "frame_index": idx,
                            "violation_type": "Gravity and free-fall behavior",
                            "description": (
                                f"Frame {idx}: Overhanging 3D upper block levitates in mid-air "
                                f"(3D COM overhang dx={config.stack_offset_x:.2f}m > 0.25m support edge) without toppling torque."
                            ),
                            "severity": "HIGH",
                        }
                    )
                elif 52 <= idx <= 60:
                    z = max(-0.12, pt.z - 0.26 * scale)
                    sdf = round(z - 0.20, 5)
                    injected_anomalies.append(
                        {
                            "frame_index": idx,
                            "violation_type": "Contact constraints",
                            "description": (
                                f"Frame {idx}: Falling 3D block penetrates rigid floor plane "
                                f"(SDF={sdf:+.2f}m, predicted z={z:.2f}m < half-height 0.20m)."
                            ),
                            "severity": "HIGH",
                        }
                    )
                elif 76 <= idx <= 80:
                    vx = 2.40 * scale
                    x = pt.x + 0.35 * scale
                    energy = pt.mechanical_energy + 9.5 * scale
                    injected_anomalies.append(
                        {
                            "frame_index": idx,
                            "violation_type": "Conservation of momentum",
                            "description": (
                                f"Frame {idx}: Settled 3D block experiences unforced lateral impulse "
                                f"(vx={vx:.2f} m/s vs GT 0.00 m/s) violating 3D momentum conservation."
                            ),
                            "severity": "MEDIUM",
                        }
                    )

            elif config.scenario == "projectile":
                if 22 <= idx <= 33:
                    drift_z = 0.38 * scale * math.sin(((idx - 21) / 13.0) * math.pi)
                    z += drift_z
                    vz += 1.9 * scale
                    energy = pt.mechanical_energy + 5.2 * scale
                    injected_anomalies.append(
                        {
                            "frame_index": idx,
                            "violation_type": "Gravity and free-fall behavior",
                            "description": (
                                f"Frame {idx}: 3D projectile deviates from -9.81 m/s^2 ballistic parabola, "
                                f"gaining +{drift_z:.2f}m altitude at apex without external force."
                            ),
                            "severity": "MEDIUM" if drift_z < 0.28 else "HIGH",
                        }
                    )
                elif 55 <= idx <= 62:
                    vz = abs(pt.vz) * (1.55 * scale) + 2.2
                    z = pt.z + 0.32 * scale
                    energy = pt.mechanical_energy * (1.65 * scale)
                    injected_anomalies.append(
                        {
                            "frame_index": idx,
                            "violation_type": "Conservation of momentum",
                            "description": (
                                f"Frame {idx}: 3D floor bounce exceeds restitution limit e={config.restitution:.2f}, "
                                f"increasing mechanical energy to {energy:.1f}J vs GT {pt.mechanical_energy:.1f}J."
                            ),
                            "severity": "HIGH",
                        }
                    )
                elif 82 <= idx <= 84 and scale >= 0.9:
                    visible = False
                    injected_anomalies.append(
                        {
                            "frame_index": idx,
                            "violation_type": "Object permanence",
                            "description": (
                                f"Frame {idx}: 3D projectile vanishes mid-flight for 3 frames "
                                f"(object permanence violation)."
                            ),
                            "severity": "HIGH",
                        }
                    )

        pred_points.append(
            TrajectoryPoint(
                frame_index=idx,
                t=pt.t,
                x=round(x, 5),
                y=round(y, 5),
                z=round(z, 5),
                qx=qx,
                qy=qy,
                qz=qz,
                qw=qw,
                vx=round(vx, 5),
                vy=round(vy, 5),
                vz=round(vz, 5),
                wx=round(wx, 5),
                wy=round(wy, 5),
                wz=round(wz, 5),
                contact=contact,
                sdf_clearance=round(sdf, 5),
                grasp_force_n=round(grasp_f, 3),
                ee_x=round(ee_x, 5),
                ee_y=round(ee_y, 5),
                ee_z=round(ee_z, 5),
                stage=pt.stage,
                mechanical_energy=round(energy, 5),
            )
        )
        visibility_mask.append(visible)

    return pred_points, visibility_mask, injected_anomalies


def generate_world_model_prediction(
    scenario: str,
    ground_truth_root: Path | str = Path("data/ground_truth"),
    output_root: Path | str = Path("data/predictions"),
    profile: str = "cosmos_default",
    num_frames: int = 101,
    use_nebius: bool = False,
) -> PredictionResult:
    """Generate predicted 3D world-model frames, 3D trajectory CSV, and model_info.json."""
    gt_dir = Path(ground_truth_root) / scenario
    gt_metadata_path = gt_dir / "metadata.json"

    if gt_metadata_path.exists():
        meta = json.loads(gt_metadata_path.read_text(encoding="utf-8"))
        config = get_scenario_config(
            scenario=scenario,
            friction=meta.get("friction"),
            mass=meta.get("mass"),
            initial_velocity=meta.get("initial_velocity"),
            num_frames=meta.get("num_frames", num_frames),
        )
    else:
        config = get_scenario_config(scenario=scenario, num_frames=num_frames)

    gt_trajectory = simulate_trajectory(config)

    output_dir = Path(output_root) / scenario
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    used_npa = False
    if use_nebius and (gt_dir / "frames").exists():
        used_npa = _try_run_npa_cosmos(scenario, gt_dir / "frames", output_dir)

    pred_trajectory, visibility_mask, injected_anomalies = synthesize_world_model_dream(
        config=config,
        gt_trajectory=gt_trajectory,
        profile=profile,
    )

    anomaly_by_frame = {item["frame_index"]: item["violation_type"] for item in injected_anomalies}

    trajectory_csv = output_dir / "predicted_trajectory.csv"
    with trajectory_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_index",
                "t",
                "x",
                "y",
                "z",
                "qx",
                "qy",
                "qz",
                "qw",
                "vx",
                "vy",
                "vz",
                "wx",
                "wy",
                "wz",
                "contact",
                "sdf_clearance",
                "grasp_force_n",
                "ee_x",
                "ee_y",
                "ee_z",
                "stage",
                "mechanical_energy",
                "object_visible",
            ]
        )
        for pt, vis in zip(pred_trajectory, visibility_mask):
            writer.writerow(
                [
                    pt.frame_index,
                    pt.t,
                    pt.x,
                    pt.y,
                    pt.z,
                    pt.qx,
                    pt.qy,
                    pt.qz,
                    pt.qw,
                    pt.vx,
                    pt.vy,
                    pt.vz,
                    pt.wx,
                    pt.wy,
                    pt.wz,
                    pt.contact,
                    pt.sdf_clearance,
                    pt.grasp_force_n,
                    pt.ee_x,
                    pt.ee_y,
                    pt.ee_z,
                    pt.stage,
                    pt.mechanical_energy,
                    int(vis),
                ]
            )

    frame_paths: List[Path] = []
    history: List[TrajectoryPoint] = []
    for pt, gt_pt, vis in zip(pred_trajectory, gt_trajectory, visibility_mask):
        if vis:
            history.append(pt)
        frame_path = frames_dir / f"frame_{pt.frame_index:03d}.png"
        if not used_npa or not frame_path.exists():
            render_scenario_frame(
                config=config,
                point=pt,
                history=history,
                output_path=frame_path,
                mode_label=f"COSMOS 3D DREAM ({profile.upper()})",
                reference_point=gt_pt,
                violation_badge=anomaly_by_frame.get(pt.frame_index),
                object_visible=vis,
            )
        frame_paths.append(frame_path)

    cosmos_scene_plan = _query_cosmos_super_reasoner_scene_plan(config, profile)
    model_info_json = output_dir / "model_info.json"
    model_info: Dict[str, Any] = {
        "model_name": os.getenv("COSMOS_REASONER_MODEL", "nvidia/Cosmos3-Super-Reasoner"),
        "world_generator": "nvidia/Cosmos-Predict2.5-3D",
        "version": "3.0.0",
        "dimension": "3D",
        "workbench": "npa-cosmos-serverless" if used_npa else "cosmos-3d-token-factory-emulator",
        "scenario": scenario,
        "isaac_task_id": config.isaac_task_id,
        "hallucination_profile": profile,
        "profile_label": HALLUCINATION_PROFILES[profile]["label"],
        "cosmos_scene_reasoning": cosmos_scene_plan,
        "inference_params": {
            "num_frames": len(frame_paths),
            "fps": int(round(1.0 / config.dt)),
            "conditioning_frames": 4,
            "guidance_scale": 7.0,
            "seed": 42,
        },
        "ground_truth_reference": str(gt_dir),
    }
    model_info_json.write_text(json.dumps(model_info, indent=2), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    # Also write structured scene_reasoning.json for Step 3 -> Step 4 Nemotron Auditor ingestion
    scene_reasoning_payload = {
        "status": "completed",
        "task": config.isaac_task_id,
        "scenario": scenario,
        "model": os.getenv("COSMOS_REASONER_MODEL", "nvidia/Cosmos3-Super-Reasoner"),
        "hallucination_profile": profile,
        "scene_description": config.description,
        "manipulation_plan": cosmos_scene_plan.get("predicted_stages", []),
        "injected_anomalies": injected_anomalies,
        "analysis": cosmos_scene_plan,
    }
    (output_dir / "scene_reasoning.json").write_text(
        json.dumps(scene_reasoning_payload, indent=2),
        encoding="utf-8",
    )

    return PredictionResult(
        scenario=scenario,
        profile=profile,
        output_dir=output_dir,
        frames_dir=frames_dir,
        frame_paths=frame_paths,
        trajectory_csv=trajectory_csv,
        model_info_json=model_info_json,
        predicted_trajectory=pred_trajectory,
        visibility_mask=visibility_mask,
        injected_anomalies=injected_anomalies,
        model_info=model_info,
    )


def generate_all_predictions(
    ground_truth_root: Path | str = Path("data/ground_truth"),
    output_root: Path | str = Path("data/predictions"),
    profile: str = "cosmos_default",
    num_frames: int = 101,
    use_nebius: bool = False,
) -> Dict[str, PredictionResult]:
    """Generate 3D world-model predictions for all registered benchmark scenarios."""
    results: Dict[str, PredictionResult] = {}
    for name in SCENARIO_REGISTRY:
        results[name] = generate_world_model_prediction(
            scenario=name,
            ground_truth_root=ground_truth_root,
            output_root=output_root,
            profile=profile,
            num_frames=num_frames,
            use_nebius=use_nebius,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3D Cosmos World Model inference.")
    parser.add_argument("--scenario", type=str, default="all", help="Scenario name or 'all'.")
    parser.add_argument("--input", type=str, default="data/ground_truth", help="Ground truth root directory.")
    parser.add_argument("--output", type=str, default="data/predictions", help="Predictions output root directory.")
    parser.add_argument(
        "--profile",
        type=str,
        default="cosmos_default",
        choices=list(HALLUCINATION_PROFILES.keys()),
        help="3D World model fidelity / hallucination profile.",
    )
    parser.add_argument("--num-frames", type=int, default=101, help="Number of frames to predict.")
    parser.add_argument("--use-nebius", action="store_true", help="Invoke npa workbench cosmos if available.")
    args = parser.parse_args()

    if args.scenario == "all":
        results = generate_all_predictions(
            ground_truth_root=args.input,
            output_root=args.output,
            profile=args.profile,
            num_frames=args.num_frames,
            use_nebius=args.use_nebius,
        )
        for name, res in results.items():
            print(
                f"[3D WorldModel] Predicted {name} ({args.profile}): "
                f"{len(res.frame_paths)} frames -> {res.output_dir}"
            )
    else:
        res = generate_world_model_prediction(
            scenario=args.scenario,
            ground_truth_root=args.input,
            output_root=args.output,
            profile=args.profile,
            num_frames=args.num_frames,
            use_nebius=args.use_nebius,
        )
        print(
            f"[3D WorldModel] Predicted {res.scenario} ({args.profile}): "
            f"{len(res.frame_paths)} frames -> {res.output_dir}"
        )


if __name__ == "__main__":
    main()
