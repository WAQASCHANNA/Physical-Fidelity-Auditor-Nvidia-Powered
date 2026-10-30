"""Prompt templates for NVIDIA Nemotron Physics Auditor (auditor/prompt_templates.py).

Implements:
1. `build_plan_audit_prompt`: Step 4 prompt comparing `ground_truth/metadata.json` against
   Cosmos `scene_reasoning.json` with Tavily real-world friction grounding.
2. `build_audit_prompt`: Frame-by-frame + scene reasoning hybrid audit prompt.
3. `build_super_escalation_prompt`: Secondary deep-reasoning verification prompt for Nemotron Super.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


SYSTEM_PROMPT = """You are a rigorous physics auditor. Your task is to compare a predicted robotic manipulation plan and video rollout against a ground-truth physical simulation and identify any violations of physical laws.

You evaluate dynamic physical consistency using SIPA (Spatial Intelligence Physical Audit) principles, isolating violations of:
- contact_constraint (Contact Constraints: objects cannot pass through or interpenetrate each other)
- object_permanence (Object Permanence: objects cannot disappear or materialize)
- gravity_support (Gravity & Support: objects cannot float or levitate without physical support/grasp closure)
- friction_slipping (Friction & Slipping: planned grasp or motion must be feasible given real-world friction coefficient)
- momentum_conservation (Conservation of Momentum: no impossible velocity or energy surges)

Always respond with a single valid JSON object adhering strictly to the requested OUTPUT FORMAT."""


def build_plan_audit_prompt(
    gt_meta: Dict[str, Any],
    cosmos_pred: Dict[str, Any],
    tavily_result: Dict[str, Any],
) -> str:
    """Build the Step 4 Nemotron Physics Auditor prompt comparing ground_truth metadata vs Cosmos scene_reasoning.json."""
    task_name = gt_meta.get("task") or gt_meta.get("isaac_task_id") or gt_meta.get("scenario", "Isaac-Lift-Cube-Franka-v0")
    object_mass = gt_meta.get("object_mass_kg") or gt_meta.get("mass", 0.10)
    friction_coefficient = gt_meta.get("friction", 0.65)
    tavily_friction_result = tavily_result.get("content") or json.dumps(tavily_result)
    cosmos_scene_reasoning_json = json.dumps(cosmos_pred, indent=2)

    return f"""SYSTEM: You are a rigorous physics auditor. Your task is to compare a predicted robotic manipulation plan against a ground-truth physical simulation and identify any violations of physical laws.

GROUND TRUTH CONTEXT (from Isaac Lab):
- Task: {task_name} (e.g., "Isaac-Lift-Cube-Franka-v0")
- Object Mass: {object_mass} kg
- Friction Coefficient: {friction_coefficient}
- Real-World Material Property (from Tavily): {tavily_friction_result}

PREDICTED SCENE REASONING (from Cosmos):
{cosmos_scene_reasoning_json}

YOUR TASK:
1. Analyze the predicted manipulation plan in the context of the ground-truth physical parameters.
2. For each step in the predicted plan, evaluate whether it violates any of the following physical constraints:
   - Contact Constraints (contact_constraint): Does the plan imply any object interpenetration (e.g., SDF < 0 or z < 0)?
   - Object Permanence (object_permanence): Does the plan assume an object disappears or materializes (object_visible == false)?
   - Gravity & Support (gravity_support): Does the plan assume an object floats or is unsupported during lift?
   - Friction & Slipping (friction_slipping): Is the planned grasp or push physically feasible given the friction coefficient (F_n >= mg / 2*mu)?
   - Conservation of Momentum (momentum_conservation): Does the plan imply an impossible transfer of momentum?

OUTPUT FORMAT (Strict JSON):
{{
  "scenario": "{task_name}",
  "overall_verdict": "PASS" | "FAIL",
  "physics_hallucination_rate": <float, 0.0 to 1.0>,
  "violations": [
    {{
      "plan_step": <int>,
      "violation_type": "contact_constraint" | "object_permanence" | "gravity_support" | "friction_slipping" | "momentum_conservation",
      "description": "A clear, specific explanation of the violation.",
      "severity": "LOW" | "MEDIUM" | "HIGH"
    }}
  ],
  "summary": "A brief, one-paragraph summary of the audit."
}}"""


def build_audit_prompt(
    scenario_name: str,
    metadata: Dict[str, Any],
    tavily_result: Dict[str, Any],
    kinematic_comparison_summary: List[Dict[str, Any]],
    cosmos_scene_reasoning: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the structured audit prompt for NVIDIA Nemotron on Nebius Token Factory."""
    task_name = metadata.get("task") or metadata.get("isaac_task_id") or scenario_name
    friction = metadata.get("friction", 0.5)
    mass = metadata.get("object_mass_kg") or metadata.get("mass", 1.0)
    initial_velocity = metadata.get("initial_velocity", 0.0)
    gravity = metadata.get("gravity", 9.81)
    restitution = metadata.get("restitution", 0.65)
    num_frames = metadata.get("num_frames") or metadata.get("steps", 101)

    tavily_snippet = tavily_result.get("content") or json.dumps(tavily_result, indent=2)
    comparison_json = json.dumps(kinematic_comparison_summary, indent=2)
    cosmos_json = json.dumps(cosmos_scene_reasoning or {}, indent=2)

    return f"""GROUND TRUTH CONTEXT (from Isaac Lab):
- Scenario / Task: {scenario_name} ({task_name})
- Object Mass: {mass} kg
- Friction Coefficient: {friction}
- Initial Velocity: {initial_velocity} m/s
- Gravity: {gravity} m/s^2
- Coefficient of Restitution: {restitution}
- Total Frames Evaluated: {num_frames}
- Real-World Material Property (from Tavily): {tavily_snippet}

PREDICTED SCENE REASONING (from Cosmos):
{cosmos_json}

FRAME-BY-FRAME 3D KINEMATIC & SDF DISCREPANCY LOG:
{comparison_json}

YOUR TASK:
Evaluate whether the predicted manipulation plan and trajectory frames violate any of the following physical laws:
- Contact constraints (contact_constraint: objects cannot pass through each other; SDF >= 0)
- Object permanence (object_permanence: objects cannot disappear)
- Gravity and free-fall behavior (gravity_support: no unsupported levitation)
- Friction and damping behavior (friction_slipping: grasp closure F_n >= mg / 2*mu and surface damping)
- Conservation of momentum (momentum_conservation: no unforced impulse or energy creation)

OUTPUT FORMAT (Strict JSON):
{{
  "scenario": "{scenario_name}",
  "overall_verdict": "PASS" | "FAIL",
  "physics_hallucination_rate": <float 0.0-1.0>,
  "violations": [
    {{
      "plan_step": <int>,
      "frame_index": <int>,
      "violation_type": "contact_constraint" | "object_permanence" | "gravity_support" | "friction_slipping" | "momentum_conservation",
      "description": "<concise quantitative description citing step/frame index and physical discrepancy>",
      "severity": "LOW" | "MEDIUM" | "HIGH"
    }}
  ],
  "summary": "<executive physics audit summary citing PHR, PIR grade, dominant failure modes, and real-world physical constant alignment>"
}}"""


def build_super_escalation_prompt(
    scenario_name: str,
    nano_report: Dict[str, Any],
    tavily_result: Dict[str, Any],
) -> str:
    """Build a secondary deep-reasoning verification prompt for Nemotron Super on boundary cases."""
    return f"""You are a Senior Physical AI Verification Judge (Nemotron Super escalation tier).
An initial pass by Nemotron 3 Nano produced the following borderline or multi-law physics audit report for scenario '{scenario_name}':

INITIAL NANO REPORT:
{json.dumps(nano_report, indent=2)}

REAL-WORLD PHYSICAL CONSTANTS (TAVILY):
{json.dumps(tavily_result, indent=2)}

Verify whether each flagged violation is physically justified under classical mechanics and refine the JSON report using the exact same OUTPUT FORMAT schema."""
