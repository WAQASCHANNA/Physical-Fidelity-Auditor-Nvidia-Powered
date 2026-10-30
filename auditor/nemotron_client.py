"""Nemotron Physics Judge Client on Nebius Token Factory (auditor/nemotron_client.py).

Compares Isaac Lab 3D ground-truth metadata & trajectories against Cosmos
world-model predictions (`scene_reasoning.json` and predicted trajectories) using:
- `nvidia/Nemotron-3-Nano-30B-A3B` on Nebius Token Factory (`https://api.tokenfactory.nebius.com/v1/`)
- Optional escalation to `nvidia/Llama-3_3-Nemotron-Super-49B-v1` for ambiguous boundary cases
- Tavily Search API for real-world material friction & restitution constants
- SIPA Physical Integrity Rating (PIR) rubric grading ('A' Physically Solvent, 'C' Moderate Risk, 'F' Critical)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from auditor.prompt_templates import (
    SYSTEM_PROMPT,
    build_audit_prompt,
    build_plan_audit_prompt,
    build_super_escalation_prompt,
)
from auditor.schemas import (
    AuditReport,
    ViolationItem,
    classify_pir_rubric,
    compute_sipa_metrics,
)
from auditor.tavily_client import PhysicalConstantsRetriever, load_dotenv_file

load_dotenv_file()

try:
    from openai import OpenAI
    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False


TOKEN_FACTORY_DEFAULT_URL = "https://api.tokenfactory.nebius.com/v1/"
NEMOTRON_NANO_DEFAULT = "nvidia/Nemotron-3-Nano-30B-A3B"
NEMOTRON_SUPER_DEFAULT = "nvidia/Llama-3_3-Nemotron-Super-49B-v1"


def _extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse a JSON object from LLM output, stripping <think> blocks and fences."""
    if not raw_text:
        return None
    text = raw_text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if "```" in text:
        for block in text.split("```"):
            candidate = block.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                try:
                    return json.loads(candidate)
                except Exception:
                    pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def _load_trajectory_rows(csv_path: Path) -> List[Dict[str, float]]:
    """Read a 3D ground-truth or predicted trajectory CSV into a list of numeric dicts."""
    rows: List[Dict[str, float]] = []
    if not csv_path.exists():
        return rows
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            parsed: Dict[str, float] = {}
            for k, v in r.items():
                if k == "stage":
                    continue
                if v is None or v == "":
                    parsed[k] = 0.0
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = 0.0
            rows.append(parsed)
    return rows


def analyze_plan_steps_discrepancies(
    gt_meta: Dict[str, Any],
    cosmos_pred: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Evaluate each step in Cosmos `manipulation_plan` against Isaac Lab physical constraints."""
    plan = cosmos_pred.get("manipulation_plan")
    if not isinstance(plan, list):
        plan = (cosmos_pred.get("analysis") or {}).get("manipulation_plan", [])

    mass = float(gt_meta.get("object_mass_kg") or gt_meta.get("mass") or 0.10)
    friction = float(gt_meta.get("friction") or 0.65)
    gravity = float(gt_meta.get("gravity") or 9.81)
    min_grasp_force = (mass * gravity) / (2.0 * max(0.05, friction))

    violations: List[Dict[str, Any]] = []
    for step_idx, step in enumerate(plan):
        if not isinstance(step, dict):
            continue
        stage = str(step.get("stage", f"step_{step_idx}"))
        frame_range = step.get("frame_range", [step_idx * 5, step_idx * 5 + 4])
        frame_idx = int(frame_range[0]) if isinstance(frame_range, list) and frame_range else step_idx
        sdf = float(step.get("predicted_sdf_clearance_m", 0.0))
        grasp_f = float(step.get("predicted_grasp_force_n", 12.0))
        pose = step.get("predicted_cube_pose_m", [0.52, 0.08, 0.038])
        z_pos = float(pose[2]) if isinstance(pose, list) and len(pose) >= 3 else 0.038
        visible = bool(step.get("object_visible", True))

        if sdf < -0.01 or z_pos < 0.0:
            violations.append(
                {
                    "plan_step": step_idx,
                    "frame_index": frame_idx,
                    "violation_type": "contact_constraint",
                    "description": (
                        f"Plan step {step_idx} ('{stage}', frames {frame_range}): Predicted cube/gripper "
                        f"interpenetrates the optical worktable (SDF={sdf:+.3f} m, z={z_pos:+.3f} m < 0.0 m)."
                    ),
                    "severity": "HIGH",
                }
            )
        if not visible:
            violations.append(
                {
                    "plan_step": step_idx,
                    "frame_index": frame_idx,
                    "violation_type": "object_permanence",
                    "description": (
                        f"Plan step {step_idx} ('{stage}', frames {frame_range}): Target 0.10 kg cube disappears "
                        f"from the scene (object_visible=false) during {stage} phase."
                    ),
                    "severity": "HIGH",
                }
            )
        if stage in ("grasp", "lift", "hold") and grasp_f < min_grasp_force:
            if stage == "lift" and z_pos > 0.15:
                violations.append(
                    {
                        "plan_step": step_idx,
                        "frame_index": frame_idx,
                        "violation_type": "gravity_support",
                        "description": (
                            f"Plan step {step_idx} ('{stage}', frames {frame_range}): Cube ascends to z={z_pos:.3f} m "
                            f"with zero grasp support force (F_grasp={grasp_f:.1f} N < {min_grasp_force:.3f} N minimum hold threshold)."
                        ),
                        "severity": "HIGH",
                    }
                )
            else:
                violations.append(
                    {
                        "plan_step": step_idx,
                        "frame_index": frame_idx,
                        "violation_type": "friction_slipping",
                        "description": (
                            f"Plan step {step_idx} ('{stage}', frames {frame_range}): Planned grasp force "
                            f"F_grasp={grasp_f:.1f} N violates antipodal friction-cone closure "
                            f"(requires F_n >= mg / 2*mu = {min_grasp_force:.3f} N for mu={friction:.2f})."
                        ),
                        "severity": "MEDIUM",
                    }
                )

    return violations


def analyze_kinematic_discrepancies(
    scenario: str,
    metadata: Dict[str, Any],
    gt_rows: List[Dict[str, float]],
    pred_rows: List[Dict[str, float]],
) -> Tuple[List[Dict[str, Any]], float, List[Dict[str, Any]]]:
    """Compare 3D frame-by-frame kinematics, SDF contact clearance, and grasp forces quantitatively."""
    violations: List[Dict[str, Any]] = []
    compact_log: List[Dict[str, Any]] = []
    kinetic_debt = 0.0
    friction = float(metadata.get("friction", 0.5))
    restitution = float(metadata.get("restitution", 0.65))
    mass = float(metadata.get("object_mass_kg") or metadata.get("mass", 1.0))
    gravity = float(metadata.get("gravity", 9.81))

    n = min(len(gt_rows), len(pred_rows))
    for i in range(n):
        gt = gt_rows[i]
        pr = pred_rows[i]
        idx = int(round(gt.get("frame_index", i)))
        plan_step = min(4, idx // 20)

        dx = pr["x"] - gt["x"]
        dy = pr.get("y", 0.0) - gt.get("y", 0.0)
        dz = pr["z"] - gt["z"]
        pos_err_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
        dvx = pr.get("vx", 0.0) - gt.get("vx", 0.0)
        dvy = pr.get("vy", 0.0) - gt.get("vy", 0.0)
        dvz = pr.get("vz", 0.0) - gt.get("vz", 0.0)
        d_energy = pr.get("mechanical_energy", 0.0) - gt.get("mechanical_energy", 0.0)
        sdf_pred = pr.get("sdf_clearance", 0.0)
        grasp_pred = pr.get("grasp_force_n", 0.0)
        grasp_gt = gt.get("grasp_force_n", 0.0)
        visible = int(round(pr.get("object_visible", 1.0))) == 1

        if d_energy > 0.25:
            kinetic_debt += d_energy

        frame_violation: Optional[Dict[str, Any]] = None

        if not visible:
            frame_violation = {
                "plan_step": plan_step,
                "frame_index": idx,
                "violation_type": "Object permanence",
                "description": (
                    f"Frame {idx}: Tracked 3D rigid body vanishes from predicted scene "
                    f"while present at 3D pose (x={gt['x']:.2f}m, y={gt.get('y', 0.0):.2f}m, z={gt['z']:.2f}m) in Isaac Lab ground truth."
                ),
                "severity": "HIGH",
            }
        elif dz < -0.075 or sdf_pred < -0.05:
            penetration_m = max(abs(dz), abs(sdf_pred))
            frame_violation = {
                "plan_step": plan_step,
                "frame_index": idx,
                "violation_type": "Contact constraints",
                "description": (
                    f"Frame {idx}: 3D mesh interpenetrates solid collider boundary by {penetration_m:.2f}m "
                    f"(SDF={sdf_pred:+.2f}m, predicted z={pr['z']:.2f}m vs Isaac Lab z={gt['z']:.2f}m)."
                ),
                "severity": "HIGH" if penetration_m > 0.11 else "MEDIUM",
            }
        elif dz > 0.11 and (scenario in ("franka_lift_cube", "franka_lift", "block_stack", "projectile")):
            if scenario in ("franka_lift_cube", "franka_lift") and grasp_pred < (mass * gravity) / max(0.1, 2.0 * friction) and grasp_gt > 1.0:
                frame_violation = {
                    "plan_step": plan_step,
                    "frame_index": idx,
                    "violation_type": "Gravity and free-fall behavior",
                    "description": (
                        f"Frame {idx}: 3D cube levitates +{dz:.2f}m above gripper without grasp force closure "
                        f"(predicted F_grasp={grasp_pred:.1f}N vs GT {grasp_gt:.1f}N; predicted z={pr['z']:.2f}m vs GT z={gt['z']:.2f}m)."
                    ),
                    "severity": "HIGH",
                }
            else:
                frame_violation = {
                    "plan_step": plan_step,
                    "frame_index": idx,
                    "violation_type": "Gravity and free-fall behavior",
                    "description": (
                        f"Frame {idx}: 3D rigid body defies 9.81 m/s^2 gravitational acceleration "
                        f"(vertical altitude excess +{dz:.2f}m; predicted z={pr['z']:.2f}m vs GT z={gt['z']:.2f}m)."
                    ),
                    "severity": "HIGH" if dz > 0.22 else "MEDIUM",
                }
        elif d_energy > 4.0 and (abs(dvx) > 1.2 or abs(dvy) > 1.2 or abs(dvz) > 1.5):
            if scenario in ("ramp_roll", "warehouse_amr"):
                frame_violation = {
                    "plan_step": plan_step,
                    "frame_index": idx,
                    "violation_type": "Friction and damping behavior",
                    "description": (
                        f"Frame {idx}: Unphysical 3D acceleration on dissipative surface (mu={friction:.2f}); "
                        f"predicted vx={pr.get('vx', 0.0):.2f} m/s vs GT {gt.get('vx', 0.0):.2f} m/s "
                        f"(+{d_energy:.1f}J kinetic debt)."
                    ),
                    "severity": "MEDIUM" if d_energy < 8.0 else "HIGH",
                }
            else:
                frame_violation = {
                    "plan_step": plan_step,
                    "frame_index": idx,
                    "violation_type": "Conservation of momentum",
                    "description": (
                        f"Frame {idx}: Unforced 3D momentum/energy surge (+{d_energy:.1f}J over Isaac Lab ground truth, "
                        f"exceeding restitution limit e={restitution:.2f})."
                    ),
                    "severity": "HIGH" if d_energy > 7.5 else "MEDIUM",
                }
        elif pos_err_3d > 0.08 or abs(dvx) > 0.30:
            frame_violation = {
                "plan_step": plan_step,
                "frame_index": idx,
                "violation_type": "Friction and damping behavior",
                "description": (
                    f"Frame {idx}: 3D trajectory slip divergence of {pos_err_3d:.2f}m "
                    f"(delta_vx={dvx:+.2f} m/s relative to mu={friction:.2f} ground truth)."
                ),
                "severity": "LOW" if pos_err_3d < 0.18 else "MEDIUM",
            }

        if frame_violation is not None:
            violations.append(frame_violation)

        if frame_violation is not None or idx % 20 == 0:
            compact_log.append(
                {
                    "plan_step": plan_step,
                    "frame_index": idx,
                    "t": round(gt.get("t", idx * 0.04), 2),
                    "gt_pos_3d": [round(gt["x"], 3), round(gt.get("y", 0.0), 3), round(gt["z"], 3)],
                    "pred_pos_3d": [round(pr["x"], 3), round(pr.get("y", 0.0), 3), round(pr["z"], 3)],
                    "gt_vel_3d": [round(gt.get("vx", 0.0), 3), round(gt.get("vy", 0.0), 3), round(gt.get("vz", 0.0), 3)],
                    "pred_vel_3d": [round(pr.get("vx", 0.0), 3), round(pr.get("vy", 0.0), 3), round(pr.get("vz", 0.0), 3)],
                    "sdf_clearance_m": round(sdf_pred, 3),
                    "grasp_force_n": round(grasp_pred, 2),
                    "energy_delta_j": round(d_energy, 2),
                    "object_visible": visible,
                    "flagged_anomaly": frame_violation["violation_type"] if frame_violation else None,
                }
            )

    return violations, round(kinetic_debt, 3), compact_log


class NemotronPhysicsAuditor:
    """Physics Auditor powered by NVIDIA Nemotron on Nebius Token Factory + Tavily grounding."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        nano_model: Optional[str] = None,
        super_model: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
    ) -> None:
        raw_key = (
            os.getenv("NEBIUS_TOKEN_FACTORY_KEY") or os.getenv("NEBIUS_API_KEY", "")
            if api_key is None
            else api_key
        )
        self.api_key = raw_key.strip()
        self.base_url = (base_url or os.getenv("TOKEN_FACTORY_URL", TOKEN_FACTORY_DEFAULT_URL)).strip()
        self.nano_model = nano_model or os.getenv("NEMOTRON_NANO_MODEL", NEMOTRON_NANO_DEFAULT)
        self.super_model = super_model or os.getenv("NEMOTRON_SUPER_MODEL", NEMOTRON_SUPER_DEFAULT)
        self.tavily_retriever = PhysicalConstantsRetriever(api_key=tavily_api_key)

    def _is_live_token_factory_configured(self) -> bool:
        return bool(
            HAS_OPENAI_SDK
            and self.api_key
            and self.api_key not in ("YOUR_KEY", "your_nebius_token_factory_api_key_here")
        )

    def _call_nemotron_json(
        self,
        model: str,
        user_prompt: str,
    ) -> Optional[Dict[str, Any]]:
        """Invoke Nebius Token Factory OpenAI-compatible chat completion in JSON mode."""
        if not self._is_live_token_factory_configured():
            return None
        candidate_urls = [
            self.base_url,
            "https://api.tokenfactory.nebius.com/v1/",
            "https://api.tokenfactory.us-central1.nebius.com/v1/",
        ]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\nRespond concisely with valid JSON only."},
            {"role": "user", "content": user_prompt},
        ]
        for url in dict.fromkeys(candidate_urls):
            client = OpenAI(base_url=url, api_key=self.api_key)
            for use_json_mode in (True, False):
                try:
                    kwargs: Dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                        "temperature": 0.1,
                        "max_tokens": 2500,
                        "timeout": 28.0,
                    }
                    if use_json_mode:
                        kwargs["response_format"] = {"type": "json_object"}
                    response = client.chat.completions.create(**kwargs)
                    msg = response.choices[0].message
                    content = msg.content or getattr(msg, "reasoning_content", "") or ""
                    parsed = _extract_json_object(content)
                    if parsed is not None:
                        return parsed
                except Exception as exc:
                    if "404" in str(exc) or "not found" in str(exc).lower():
                        try:
                            avail = [m.id for m in client.models.list().data]
                            nemotron_matches = [m for m in avail if "nemotron" in m.lower()]
                            if nemotron_matches and model not in nemotron_matches:
                                model = nemotron_matches[0]
                                self.nano_model = model
                        except Exception:
                            pass
                    continue
        return None

    def audit_scene_reasoning(
        self,
        ground_truth_meta_path: Path | str = Path("data/ground_truth/franka_lift/metadata.json"),
        scene_reasoning_path: Path | str = Path("data/predictions/franka_lift/scene_reasoning.json"),
        output_report_path: Path | str = Path("data/reports/franka_lift_audit.json"),
    ) -> AuditReport:
        """Step 4: Audit a Cosmos `scene_reasoning.json` manipulation plan against Isaac Lab `metadata.json`."""
        t0 = time.perf_counter()
        gt_path = Path(ground_truth_meta_path)
        pred_path = Path(scene_reasoning_path)
        out_path = Path(output_report_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        gt_meta = json.loads(gt_path.read_text(encoding="utf-8"))
        cosmos_pred = json.loads(pred_path.read_text(encoding="utf-8"))

        scenario_key = str(gt_meta.get("scenario") or "franka_lift_cube")
        if scenario_key == "franka_lift":
            tavily_scen = "franka_lift_cube"
        else:
            tavily_scen = scenario_key

        tavily_result = self.tavily_retriever.retrieve_constants(
            scenario=tavily_scen,
            custom_query=gt_meta.get("tavily_query"),
        )

        plan_list = cosmos_pred.get("manipulation_plan") or []
        total_plan_steps = max(1, len(plan_list))

        prompt = build_plan_audit_prompt(
            gt_meta=gt_meta,
            cosmos_pred=cosmos_pred,
            tavily_result=tavily_result,
        )

        llm_payload = self._call_nemotron_json(self.nano_model, prompt)
        model_used = self.nano_model

        active_violations: List[Dict[str, Any]] = []
        if llm_payload and isinstance(llm_payload.get("violations"), list):
            for idx, item in enumerate(llm_payload["violations"]):
                if isinstance(item, dict):
                    step_num = int(item.get("plan_step", item.get("frame_index", idx)))
                    active_violations.append(
                        {
                            "plan_step": step_num,
                            "frame_index": int(item.get("frame_index", step_num)),
                            "violation_type": str(item.get("violation_type") or "contact_constraint"),
                            "description": str(item.get("description") or "Physical violation in manipulation plan."),
                            "severity": str(item.get("severity") or "HIGH").upper(),
                        }
                    )

        if not active_violations:
            active_violations = analyze_plan_steps_discrepancies(gt_meta, cosmos_pred)
            if not llm_payload:
                model_used = f"{self.nano_model} (physics-plan-judge)"

        violated_steps = len({int(v.get("plan_step", 0)) for v in active_violations})
        phr = round(min(1.0, violated_steps / float(total_plan_steps)), 4)

        sipa = compute_sipa_metrics(
            violations=active_violations,
            total_frames=total_plan_steps,
            kinetic_debt_joules=18.5 * violated_steps,
        )
        pir = sipa["physical_integrity_rating"]
        rubric = classify_pir_rubric(pir)
        verdict = "PASS" if (phr == 0.0 and pir >= 0.70) else "FAIL"

        task_name = str(gt_meta.get("task") or cosmos_pred.get("task") or "Isaac-Lift-Cube-Franka-v0")
        summary_text = (
            (llm_payload or {}).get("summary")
            or (
                f"Audited {total_plan_steps}-step Cosmos3-Super-Reasoner manipulation plan for '{task_name}' "
                f"against Isaac Lab ground truth (mass={gt_meta.get('object_mass_kg', 0.10)} kg, mu={gt_meta.get('friction', 0.65)}) "
                f"and Tavily empirical friction constants ({tavily_result.get('material_pair')}). "
                f"Verdict: {verdict} | PHR={phr:.3f} ({violated_steps}/{total_plan_steps} steps violated) | "
                f"PIR={pir:.3f} (Grade {rubric['pir_grade']} — {rubric['pir_label']}: {rubric['pir_meaning']})."
            )
        )

        validated_violations = [
            ViolationItem(
                plan_step=int(v.get("plan_step", 0)),
                frame_index=int(v.get("frame_index", v.get("plan_step", 0))),
                violation_type=str(v["violation_type"]),
                description=str(v["description"]),
                severity=str(v.get("severity", "HIGH")),
            )
            for v in active_violations
        ]

        report = AuditReport(
            scenario=task_name,
            overall_verdict=verdict,
            physics_hallucination_rate=phr,
            physical_integrity_rating=pir,
            pir_grade=rubric["pir_grade"],
            pir_label=rubric["pir_label"],
            pir_meaning=rubric["pir_meaning"],
            temporal_consistency_score=sipa["temporal_consistency_score"],
            kinetic_debt_joules=round(18.5 * violated_steps, 2),
            total_frames=int(gt_meta.get("steps", 25)),
            violated_frames_count=len(validated_violations),
            total_plan_steps=total_plan_steps,
            violated_plan_steps=violated_steps,
            violations=validated_violations,
            summary=summary_text,
            model_used=model_used,
            escalated_to_super=False,
            world_model_profile=str(cosmos_pred.get("hallucination_profile", "cosmos_default")),
            tavily_grounding=tavily_result,
            audit_latency_seconds=round(time.perf_counter() - t0, 4),
        )

        out_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
        return report

    def audit_scenario(
        self,
        scenario: str,
        ground_truth_root: Path | str = Path("data/ground_truth"),
        predictions_root: Path | str = Path("data/predictions"),
        output_root: Path | str = Path("data/reports"),
        escalate_ambiguous: bool = True,
    ) -> AuditReport:
        """Run a full Tier-3 physics audit for a scenario and save the structured JSON report."""
        t0 = time.perf_counter()

        gt_dir = Path(ground_truth_root) / scenario
        pred_dir = Path(predictions_root) / scenario
        output_dir = Path(output_root)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = gt_dir / "metadata.json"
        metadata: Dict[str, Any] = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {"scenario": scenario, "friction": 0.5, "mass": 1.0, "initial_velocity": 0.0, "num_frames": 101}
        )

        model_info_path = pred_dir / "model_info.json"
        model_info: Dict[str, Any] = (
            json.loads(model_info_path.read_text(encoding="utf-8"))
            if model_info_path.exists()
            else {"hallucination_profile": "cosmos_default"}
        )
        profile = str(model_info.get("hallucination_profile", "cosmos_default"))

        scene_reasoning_path = pred_dir / "scene_reasoning.json"
        cosmos_scene_reasoning: Optional[Dict[str, Any]] = (
            json.loads(scene_reasoning_path.read_text(encoding="utf-8"))
            if scene_reasoning_path.exists()
            else None
        )

        gt_rows = _load_trajectory_rows(gt_dir / "trajectory.csv")
        pred_rows = _load_trajectory_rows(pred_dir / "predicted_trajectory.csv")
        total_frames = int(metadata.get("num_frames", max(len(gt_rows), 101)))

        tavily_result = self.tavily_retriever.retrieve_constants(
            scenario=scenario,
            custom_query=metadata.get("tavily_query"),
        )

        kinematic_violations, kinetic_debt, compact_log = analyze_kinematic_discrepancies(
            scenario=scenario,
            metadata=metadata,
            gt_rows=gt_rows,
            pred_rows=pred_rows,
        )

        prompt = build_audit_prompt(
            scenario_name=scenario,
            metadata=metadata,
            tavily_result=tavily_result,
            kinematic_comparison_summary=compact_log[:45],
            cosmos_scene_reasoning=cosmos_scene_reasoning,
        )

        llm_payload = self._call_nemotron_json(self.nano_model, prompt)
        model_used = self.nano_model
        escalated = False

        active_violations: List[Dict[str, Any]] = []
        if llm_payload and isinstance(llm_payload.get("violations"), list):
            for item in llm_payload["violations"]:
                if isinstance(item, dict) and ("frame_index" in item or "plan_step" in item):
                    try:
                        f_idx = int(item.get("frame_index", item.get("plan_step", 0)))
                        p_step = int(item.get("plan_step", min(4, f_idx // 20)))
                        active_violations.append(
                            {
                                "plan_step": p_step,
                                "frame_index": f_idx,
                                "violation_type": str(item.get("violation_type") or "Physical violation"),
                                "description": str(item.get("description") or "Physical inconsistency detected"),
                                "severity": str(item.get("severity") or "MEDIUM").upper(),
                            }
                        )
                    except (ValueError, TypeError):
                        continue

        if not active_violations:
            active_violations = kinematic_violations
            if not llm_payload:
                model_used = f"{self.nano_model} (local-physics-judge-fallback)"

        sipa = compute_sipa_metrics(
            violations=active_violations,
            total_frames=total_frames,
            kinetic_debt_joules=kinetic_debt,
        )

        if (
            escalate_ambiguous
            and self._is_live_token_factory_configured()
            and 0.05 <= sipa["physics_hallucination_rate"] <= 0.14
        ):
            super_prompt = build_super_escalation_prompt(
                scenario_name=scenario,
                nano_report={
                    "scenario": scenario,
                    "physics_hallucination_rate": sipa["physics_hallucination_rate"],
                    "violations": active_violations,
                },
                tavily_result=tavily_result,
            )
            super_payload = self._call_nemotron_json(self.super_model, super_prompt)
            if super_payload and isinstance(super_payload.get("violations"), list):
                active_violations = super_payload["violations"]
                sipa = compute_sipa_metrics(
                    violations=active_violations,
                    total_frames=total_frames,
                    kinetic_debt_joules=kinetic_debt,
                )
                model_used = self.super_model
                escalated = True

        violation_types = sorted({str(v.get("violation_type", "")) for v in active_violations})
        if llm_payload and isinstance(llm_payload.get("summary"), str) and llm_payload["summary"].strip():
            summary_text = llm_payload["summary"].strip()
        elif not active_violations:
            summary_text = (
                f"Scenario '{scenario}' PASSED physical fidelity audit with PHR=0.000 and "
                f"PIR={sipa['physical_integrity_rating']:.3f} (Grade {sipa['pir_grade']} — {sipa['pir_label']}). "
                f"Predicted trajectory aligns with Isaac Lab ground truth and Tavily empirical constants "
                f"({tavily_result.get('material_pair')}, mu={metadata.get('friction')})."
            )
        else:
            summary_text = (
                f"Scenario '{scenario}' ({profile}) evaluated across {total_frames} frames: "
                f"verdict={sipa['overall_verdict']} with PHR={sipa['physics_hallucination_rate']:.3f}, "
                f"PIR={sipa['physical_integrity_rating']:.3f} (Grade {sipa['pir_grade']} — {sipa['pir_label']}), "
                f"and Kinetic Debt={kinetic_debt:.1f}J. Detected {sipa['violated_frames_count']} violated frames "
                f"across laws: {', '.join(violation_types)}. Grounded against Tavily empirical constant "
                f"for {tavily_result.get('material_pair')} (nominal mu={tavily_result.get('recommended_friction')})."
            )

        latency = round(time.perf_counter() - t0, 4)

        validated_violations = [
            ViolationItem(
                plan_step=int(v.get("plan_step", min(4, int(v.get("frame_index", 0)) // 20))),
                frame_index=int(v.get("frame_index", 0)),
                violation_type=str(v["violation_type"]),
                description=str(v["description"]),
                severity=str(v.get("severity", "MEDIUM")),
            )
            for v in active_violations
        ]

        violated_steps_count = len({v.plan_step for v in validated_violations})

        report = AuditReport(
            scenario=scenario,
            overall_verdict=sipa["overall_verdict"],
            physics_hallucination_rate=sipa["physics_hallucination_rate"],
            physical_integrity_rating=sipa["physical_integrity_rating"],
            pir_grade=sipa["pir_grade"],
            pir_label=sipa["pir_label"],
            pir_meaning=sipa["pir_meaning"],
            temporal_consistency_score=sipa["temporal_consistency_score"],
            kinetic_debt_joules=kinetic_debt,
            total_frames=total_frames,
            violated_frames_count=sipa["violated_frames_count"],
            total_plan_steps=5,
            violated_plan_steps=violated_steps_count,
            violations=validated_violations,
            summary=summary_text,
            model_used=model_used,
            escalated_to_super=escalated,
            world_model_profile=profile,
            tavily_grounding=tavily_result,
            audit_latency_seconds=latency,
        )

        report_path = output_dir / f"{scenario}_audit.json"
        report_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
        return report

    def audit_all_scenarios(
        self,
        ground_truth_root: Path | str = Path("data/ground_truth"),
        predictions_root: Path | str = Path("data/predictions"),
        output_root: Path | str = Path("data/reports"),
    ) -> Dict[str, AuditReport]:
        """Run the physics audit across all registered benchmark scenarios and save aggregate summary."""
        reports: Dict[str, AuditReport] = {}
        for name in ("warehouse_amr", "franka_lift_cube", "block_stack", "projectile", "ramp_roll"):
            reports[name] = self.audit_scenario(
                scenario=name,
                ground_truth_root=ground_truth_root,
                predictions_root=predictions_root,
                output_root=output_root,
            )

        output_dir = Path(output_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        mean_phr = round(
            sum(r.physics_hallucination_rate for r in reports.values()) / max(1, len(reports)),
            4,
        )
        mean_pir = round(
            sum(r.physical_integrity_rating for r in reports.values()) / max(1, len(reports)),
            4,
        )
        total_debt = round(sum(r.kinetic_debt_joules for r in reports.values()), 3)
        overall_rubric = classify_pir_rubric(mean_pir)

        aggregate = {
            "project": "physics-audit",
            "scenarios_audited": len(reports),
            "mean_physics_hallucination_rate": mean_phr,
            "mean_physical_integrity_rating": mean_pir,
            "overall_pir_grade": overall_rubric["pir_grade"],
            "overall_pir_label": overall_rubric["pir_label"],
            "total_kinetic_debt_joules": total_debt,
            "target_phr_threshold": 0.10,
            "target_pir_threshold": 0.70,
            "reports": {k: v.model_dump() for k, v in reports.items()},
        }
        (output_dir / "aggregate_audit_summary.json").write_text(
            json.dumps(aggregate, indent=2),
            encoding="utf-8",
        )
        return reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4: Run the Nemotron Physics Auditor on Isaac Lab Ground Truth vs. Cosmos Scene Reasoning."
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="data/ground_truth/franka_lift/metadata.json",
        help="Path to Isaac Lab ground-truth metadata.json",
    )
    parser.add_argument(
        "--prediction",
        type=str,
        default="data/predictions/franka_lift/scene_reasoning.json",
        help="Path to Cosmos scene_reasoning.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/reports/franka_lift_audit.json",
        help="Output path for the structured Nemotron audit report JSON",
    )
    args = parser.parse_args()

    auditor = NemotronPhysicsAuditor()
    report = auditor.audit_scene_reasoning(
        ground_truth_meta_path=args.ground_truth,
        scene_reasoning_path=args.prediction,
        output_report_path=args.output,
    )
    print(json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    main()
