"""Unit and integration test suite for Physical Fidelity Auditor (tests/test_scenarios.py).

Verifies:
1. Module A (ground_truth): physical determinism, energy/contact behavior, and output contracts.
2. Module B (world_model): prediction frame generation, model_info.json, and hallucination profiles.
3. Module C (auditor): Tavily constant retrieval, prompt construction, SIPA PHR/PIR math, and JSON schema validation.
4. End-to-End Pipeline: full 3-tier execution for both hallucinating and physically solvent profiles.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auditor.nemotron_client import NemotronPhysicsAuditor
from auditor.prompt_templates import build_audit_prompt, build_plan_audit_prompt
from auditor.schemas import classify_pir_rubric, compute_sipa_metrics
from auditor.tavily_client import PhysicalConstantsRetriever
from ground_truth.scenarios import (
    SCENARIO_REGISTRY,
    generate_ground_truth,
    get_scenario_config,
    simulate_trajectory,
)
from npa.workbench.token_factory import reason_scene
from pipeline import run_scenario_pipeline
from world_model.predictor import generate_world_model_prediction


class TestPhysicalFidelityAuditor(unittest.TestCase):
    """Test suite for the physics-audit repository."""

    def test_scenario_registry_coverage(self) -> None:
        """All four 3D scenarios (including Franka Lift Cube) must be registered with valid parameters."""
        for expected in ("franka_lift_cube", "block_stack", "projectile", "ramp_roll"):
            self.assertIn(expected, SCENARIO_REGISTRY)
            cfg = get_scenario_config(expected, num_frames=25)
            self.assertEqual(cfg.scenario, expected)
            self.assertGreater(cfg.mass, 0.0)
            self.assertGreaterEqual(cfg.friction, 0.0)
            self.assertEqual(cfg.num_frames, 25)

    def test_franka_lift_cube_3d_manipulation(self) -> None:
        """Franka arm cube lift (Isaac-Lift-Cube-Franka-v0) must grasp and lift the cube in 3D space."""
        pts = simulate_trajectory(get_scenario_config("franka_lift_cube", num_frames=101))
        self.assertEqual(pts[0].stage, "approach")
        self.assertEqual(pts[42].stage, "grasp")
        self.assertEqual(pts[-1].stage, "hold")
        self.assertGreater(pts[-1].z, pts[0].z + 0.5)
        self.assertGreater(pts[-1].grasp_force_n, 5.0)
        self.assertGreaterEqual(min(p.sdf_clearance for p in pts), 0.0)

    def test_ground_truth_physics_and_output_contract(self) -> None:
        """Ground truth generator must create trajectory.csv, metadata.json, and PNG frames."""
        with tempfile.TemporaryDirectory() as tmp:
            res = generate_ground_truth(
                scenario="ramp_roll",
                output_root=Path(tmp) / "ground_truth",
                num_frames=21,
            )
            self.assertTrue(res.trajectory_csv.exists())
            self.assertTrue(res.metadata_json.exists())
            self.assertEqual(len(res.frame_paths), 21)
            self.assertTrue(res.frame_paths[0].exists())
            # Verify PNG magic bytes
            header = res.frame_paths[0].read_bytes()[:8]
            self.assertEqual(header, b"\x89PNG\r\n\x1a\n")
            # Sphere rolling down ramp must advance in +x and descend in z
            self.assertGreater(res.trajectory[-1].x, res.trajectory[0].x)
            self.assertLessEqual(res.trajectory[-1].z, res.trajectory[0].z)

    def test_block_stack_and_projectile_kinematics(self) -> None:
        """Block stack overhang must topple downward; projectile must follow gravity and floor bounce."""
        stack_pts = simulate_trajectory(get_scenario_config("block_stack", num_frames=60))
        self.assertLess(stack_pts[-1].z, stack_pts[0].z)
        self.assertTrue( all(pt.z >= 0.19 for pt in stack_pts) )

        proj_pts = simulate_trajectory(get_scenario_config("projectile", num_frames=60))
        max_z = max(pt.z for pt in proj_pts)
        self.assertGreater(max_z, proj_pts[0].z)
        self.assertTrue(all(pt.z >= 0.15 for pt in proj_pts))

    def test_world_model_prediction_output_contract(self) -> None:
        """World model predictor must output predicted frames and model_info.json."""
        with tempfile.TemporaryDirectory() as tmp:
            gt_root = Path(tmp) / "ground_truth"
            pred_root = Path(tmp) / "predictions"
            generate_ground_truth("projectile", output_root=gt_root, num_frames=30)
            pred = generate_world_model_prediction(
                scenario="projectile",
                ground_truth_root=gt_root,
                output_root=pred_root,
                profile="cosmos_default",
                num_frames=30,
            )
            self.assertTrue(pred.model_info_json.exists())
            self.assertTrue(pred.trajectory_csv.exists())
            self.assertEqual(len(pred.frame_paths), 30)
            self.assertGreater(len(pred.injected_anomalies), 0)

    def test_tavily_retriever_and_prompt_builder(self) -> None:
        """Tavily retriever must return physical constants and cache them; prompt must include JSON schema."""
        with tempfile.TemporaryDirectory() as tmp:
            cache_file = Path(tmp) / "tavily_cache.json"
            retriever = PhysicalConstantsRetriever(api_key="", cache_path=cache_file)
            constants = retriever.retrieve_constants("ramp_roll")
            self.assertEqual(constants["recommended_friction"], 0.50)
            self.assertTrue(cache_file.exists())

            prompt = build_audit_prompt(
                scenario_name="ramp_roll",
                metadata={"friction": 0.5, "mass": 1.0, "initial_velocity": 0.2},
                tavily_result=constants,
                kinematic_comparison_summary=[{"frame_index": 30, "flagged_anomaly": "Contact constraints"}],
            )
            self.assertIn("physics_hallucination_rate", prompt)
            self.assertIn("Contact constraints", prompt)

    def test_sipa_metrics_pass_and_fail_thresholds(self) -> None:
        """Physically solvent models must achieve PHR < 0.10, PIR > 0.85, and PASS verdict."""
        solvent = compute_sipa_metrics(
            violations=[
                {
                    "frame_index": 54,
                    "violation_type": "Friction and damping behavior",
                    "description": "Mild slip",
                    "severity": "LOW",
                }
            ],
            total_frames=101,
            kinetic_debt_joules=0.2,
        )
        self.assertLess(solvent["physics_hallucination_rate"], 0.10)
        self.assertGreater(solvent["physical_integrity_rating"], 0.85)
        self.assertEqual(solvent["overall_verdict"], "PASS")

        hallucinated = compute_sipa_metrics(
            violations=[
                {
                    "frame_index": i,
                    "violation_type": "Contact constraints",
                    "description": "Ramp interpenetration",
                    "severity": "HIGH",
                }
                for i in range(28, 45)
            ],
            total_frames=101,
            kinetic_debt_joules=42.0,
        )
        self.assertGreater(hallucinated["physics_hallucination_rate"], 0.10)
        self.assertEqual(hallucinated["overall_verdict"], "FAIL")

    def test_end_to_end_pipeline_both_profiles(self) -> None:
        """End-to-end pipeline must produce FAIL on hallucinating profile and PASS on solvent profile."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"

            fail_bundle = run_scenario_pipeline(
                scenario="ramp_roll",
                profile="cosmos_default",
                data_dir=data_dir,
                num_frames=101,
            )
            fail_rep = fail_bundle["audit_report"]
            self.assertEqual(fail_rep.overall_verdict, "FAIL")
            self.assertGreater(fail_rep.physics_hallucination_rate, 0.10)

            pass_bundle = run_scenario_pipeline(
                scenario="ramp_roll",
                profile="physically_solvent",
                data_dir=data_dir,
                num_frames=101,
            )
            pass_rep = pass_bundle["audit_report"]
            self.assertEqual(pass_rep.overall_verdict, "PASS")
            self.assertLess(pass_rep.physics_hallucination_rate, 0.10)
            self.assertGreaterEqual(pass_rep.physical_integrity_rating, 0.85)

            report_file = data_dir / "reports" / "ramp_roll_audit.json"
            self.assertTrue(report_file.exists())
            parsed = json.loads(report_file.read_text(encoding="utf-8"))
            self.assertEqual(parsed["scenario"], "ramp_roll")

    def test_cosmos_token_factory_reason_scene_contract(self) -> None:
        """Step 3 reason_scene SDK must write scene_reasoning.json and metadata.json."""
        with tempfile.TemporaryDirectory() as tmp:
            gt_root = Path(tmp) / "ground_truth"
            pred_out = Path(tmp) / "predictions" / "franka_lift"
            generate_ground_truth("franka_lift_cube", output_root=gt_root, num_frames=10)
            res = reason_scene(
                input_path=gt_root / "franka_lift_cube" / "frames",
                output_path=pred_out,
                model="nvidia/Cosmos3-Super-Reasoner",
            )
            self.assertEqual(res["status"], "completed")
            self.assertTrue((pred_out / "scene_reasoning.json").exists())
            self.assertTrue((pred_out / "metadata.json").exists())
            self.assertIn("scene_description", res)
            self.assertIn("manipulation_plan", res)

    def test_step4_nemotron_scene_reasoning_auditor_and_pir_rubric(self) -> None:
        """Step 4 audit_scene_reasoning must evaluate manipulation_plan and compute PHR + SIPA PIR rubric."""
        grade_a = classify_pir_rubric(0.85)
        self.assertEqual(grade_a["pir_grade"], "A")
        self.assertEqual(grade_a["pir_label"], "Physically Solvent")

        grade_c = classify_pir_rubric(0.55)
        self.assertEqual(grade_c["pir_grade"], "C")
        self.assertEqual(grade_c["pir_label"], "Moderate Risk")

        grade_f = classify_pir_rubric(0.15)
        self.assertEqual(grade_f["pir_grade"], "F")
        self.assertEqual(grade_f["pir_label"], "Critical")

        with tempfile.TemporaryDirectory() as tmp:
            gt_root = Path(tmp) / "ground_truth"
            pred_out = Path(tmp) / "predictions" / "franka_lift"
            report_out = Path(tmp) / "reports" / "franka_lift_audit.json"

            gt_res = generate_ground_truth("franka_lift_cube", output_root=gt_root, num_frames=10)
            reason_scene(
                input_path=gt_root / "franka_lift_cube" / "frames",
                output_path=pred_out,
                model="nvidia/Cosmos3-Super-Reasoner",
            )
            gt_meta = json.loads(gt_res.metadata_json.read_text(encoding="utf-8"))
            cosmos_pred = json.loads((pred_out / "scene_reasoning.json").read_text(encoding="utf-8"))

            prompt = build_plan_audit_prompt(
                gt_meta=gt_meta,
                cosmos_pred=cosmos_pred,
                tavily_result={"recommended_friction": 0.65, "material_pair": "rubber on ABS"},
            )
            self.assertIn("GROUND TRUTH CONTEXT (from Isaac Lab):", prompt)
            self.assertIn("PREDICTED SCENE REASONING (from Cosmos):", prompt)
            self.assertIn("contact_constraint", prompt)

            auditor = NemotronPhysicsAuditor(api_key="", tavily_api_key="")
            report = auditor.audit_scene_reasoning(
                ground_truth_meta_path=gt_res.metadata_json,
                scene_reasoning_path=pred_out / "scene_reasoning.json",
                output_report_path=report_out,
            )
            self.assertTrue(report_out.exists())
            self.assertEqual(report.overall_verdict, "FAIL")
            self.assertGreater(report.physics_hallucination_rate, 0.0)
            self.assertIn(report.pir_grade, ("A", "C", "F"))
            self.assertGreater(len(report.violations), 0)


if __name__ == "__main__":
    unittest.main()

