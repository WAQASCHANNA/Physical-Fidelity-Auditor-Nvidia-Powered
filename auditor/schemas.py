"""Schema definitions and SIPA metric calculators for the Physics Auditor (auditor/schemas.py).

Validates Nemotron JSON audit outputs using Pydantic and computes:
- Physics Hallucination Rate (PHR): violated_steps / total_steps (or violated_frames / total_frames)
- Physical Integrity Rating (PIR) & SIPA Rubric Grade:
    * PIR >= 0.70       -> Grade 'A' ('Physically Solvent': High physical integrity; reliable under stress)
    * 0.30 <= PIR < 0.70 -> Grade 'C' ('Moderate Risk': Some physical debt; may fail in novel situations)
    * PIR < 0.30        -> Grade 'F' ('Critical': Physical bankruptcy; trajectory violates fundamental causality)
- Kinetic Debt (Joules): cumulative unphysical mechanical energy violation across steps/frames
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

try:
    from pydantic import BaseModel, Field, field_validator
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object  # type: ignore[misc,assignment]


SeverityLevel = Literal["LOW", "MEDIUM", "HIGH"]
VerdictType = Literal["PASS", "FAIL"]
PIRGradeType = Literal["A", "C", "F"]

SEVERITY_WEIGHTS: Dict[str, float] = {
    "LOW": 0.35,
    "MEDIUM": 0.70,
    "HIGH": 1.00,
}


def classify_pir_rubric(pir: float) -> Dict[str, str]:
    """Map a Physical Integrity Rating (PIR in [0, 1]) to its SIPA Rubric Grade, Label, and Meaning."""
    if pir >= 0.70:
        return {
            "pir_grade": "A",
            "pir_label": "Physically Solvent",
            "pir_meaning": "High physical integrity; reliable under stress.",
        }
    if pir >= 0.30:
        return {
            "pir_grade": "C",
            "pir_label": "Moderate Risk",
            "pir_meaning": "Some physical debt; may fail in novel situations.",
        }
    return {
        "pir_grade": "F",
        "pir_label": "Critical",
        "pir_meaning": "Physical bankruptcy; trajectory violates fundamental causality.",
    }


if HAS_PYDANTIC:

    class ViolationItem(BaseModel):
        """Single plan-step or frame-level physical law violation identified during audit."""

        plan_step: int = Field(default=0, ge=0, description="Plan step index in manipulation_plan")
        frame_index: int = Field(default=0, ge=0, description="Zero-based frame index where violation occurs")
        violation_type: str = Field(..., min_length=2, description="Physical law or constraint violated")
        description: str = Field(..., min_length=3, description="Clear, specific explanation of the violation")
        severity: SeverityLevel = Field(default="MEDIUM", description="Violation severity: LOW, MEDIUM, or HIGH")

        @field_validator("severity", mode="before")
        @classmethod
        def _normalize_severity(cls, v: Any) -> str:
            val = str(v).strip().upper()
            return val if val in ("LOW", "MEDIUM", "HIGH") else "MEDIUM"

    class AuditReport(BaseModel):
        """Validated JSON audit report produced by the Nemotron Physics Auditor."""

        scenario: str
        overall_verdict: VerdictType
        physics_hallucination_rate: float = Field(..., ge=0.0, le=1.0)
        physical_integrity_rating: float = Field(default=1.0, ge=0.0, le=1.0)
        pir_grade: PIRGradeType = Field(default="A")
        pir_label: str = Field(default="Physically Solvent")
        pir_meaning: str = Field(default="High physical integrity; reliable under stress.")
        temporal_consistency_score: float = Field(default=1.0, ge=0.0, le=1.0)
        kinetic_debt_joules: float = Field(default=0.0, ge=0.0)
        total_frames: int = Field(default=101, ge=1)
        violated_frames_count: int = Field(default=0, ge=0)
        total_plan_steps: int = Field(default=5, ge=1)
        violated_plan_steps: int = Field(default=0, ge=0)
        violations: List[ViolationItem] = Field(default_factory=list)
        summary: str
        model_used: str = Field(default="nvidia/Nemotron-3-Nano-30B-A3B")
        escalated_to_super: bool = Field(default=False)
        world_model_profile: str = Field(default="cosmos_default")
        tavily_grounding: Dict[str, Any] = Field(default_factory=dict)
        audit_latency_seconds: float = Field(default=0.0, ge=0.0)

else:

    class ViolationItem:  # type: ignore[no-redef]
        def __init__(
            self,
            violation_type: str,
            description: str,
            severity: str = "MEDIUM",
            frame_index: int = 0,
            plan_step: Optional[int] = None,
        ) -> None:
            self.frame_index = max(0, int(frame_index))
            self.plan_step = max(0, int(plan_step if plan_step is not None else frame_index))
            self.violation_type = str(violation_type)
            self.description = str(description)
            sev = str(severity).strip().upper()
            self.severity = sev if sev in ("LOW", "MEDIUM", "HIGH") else "MEDIUM"

        def model_dump(self) -> Dict[str, Any]:
            return {
                "plan_step": self.plan_step,
                "frame_index": self.frame_index,
                "violation_type": self.violation_type,
                "description": self.description,
                "severity": self.severity,
            }

    class AuditReport:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            self.scenario = str(kwargs["scenario"])
            self.overall_verdict = "PASS" if str(kwargs.get("overall_verdict", "FAIL")).upper() == "PASS" else "FAIL"
            self.physics_hallucination_rate = float(kwargs.get("physics_hallucination_rate", 0.0))
            self.physical_integrity_rating = float(kwargs.get("physical_integrity_rating", 1.0))
            rubric = classify_pir_rubric(self.physical_integrity_rating)
            self.pir_grade = str(kwargs.get("pir_grade", rubric["pir_grade"]))
            self.pir_label = str(kwargs.get("pir_label", rubric["pir_label"]))
            self.pir_meaning = str(kwargs.get("pir_meaning", rubric["pir_meaning"]))
            self.temporal_consistency_score = float(kwargs.get("temporal_consistency_score", 1.0))
            self.kinetic_debt_joules = float(kwargs.get("kinetic_debt_joules", 0.0))
            self.total_frames = int(kwargs.get("total_frames", 101))
            self.violated_frames_count = int(kwargs.get("violated_frames_count", 0))
            self.total_plan_steps = int(kwargs.get("total_plan_steps", 5))
            self.violated_plan_steps = int(kwargs.get("violated_plan_steps", 0))
            raw_v = kwargs.get("violations", [])
            self.violations = [
                v if isinstance(v, ViolationItem) else ViolationItem(**v) for v in raw_v
            ]
            self.summary = str(kwargs.get("summary", ""))
            self.model_used = str(kwargs.get("model_used", "nvidia/Nemotron-3-Nano-30B-A3B"))
            self.escalated_to_super = bool(kwargs.get("escalated_to_super", False))
            self.world_model_profile = str(kwargs.get("world_model_profile", "cosmos_default"))
            self.tavily_grounding = dict(kwargs.get("tavily_grounding", {}))
            self.audit_latency_seconds = float(kwargs.get("audit_latency_seconds", 0.0))

        def model_dump(self) -> Dict[str, Any]:
            return {
                "scenario": self.scenario,
                "overall_verdict": self.overall_verdict,
                "physics_hallucination_rate": self.physics_hallucination_rate,
                "physical_integrity_rating": self.physical_integrity_rating,
                "pir_grade": self.pir_grade,
                "pir_label": self.pir_label,
                "pir_meaning": self.pir_meaning,
                "temporal_consistency_score": self.temporal_consistency_score,
                "kinetic_debt_joules": self.kinetic_debt_joules,
                "total_frames": self.total_frames,
                "violated_frames_count": self.violated_frames_count,
                "total_plan_steps": self.total_plan_steps,
                "violated_plan_steps": self.violated_plan_steps,
                "violations": [v.model_dump() for v in self.violations],
                "summary": self.summary,
                "model_used": self.model_used,
                "escalated_to_super": self.escalated_to_super,
                "world_model_profile": self.world_model_profile,
                "tavily_grounding": self.tavily_grounding,
                "audit_latency_seconds": self.audit_latency_seconds,
            }


def compute_sipa_metrics(
    violations: List[Dict[str, Any]],
    total_frames: int,
    kinetic_debt_joules: float = 0.0,
) -> Dict[str, Any]:
    """Compute PHR, PIR, SIPA Rubric Grade (A / C / F), temporal consistency, and PASS/FAIL verdict."""
    total_frames = max(1, int(total_frames))
    unique_frames = sorted(
        {
            int(v.get("frame_index", v.get("plan_step", 0)))
            for v in violations
            if 0 <= int(v.get("frame_index", v.get("plan_step", 0))) < total_frames
        }
    )
    violated_count = len(unique_frames)
    phr = round(min(1.0, violated_count / float(total_frames)), 4)

    if not violations:
        weighted_severity_penalty = 0.0
    else:
        severity_sum = sum(
            SEVERITY_WEIGHTS.get(str(v.get("severity", "MEDIUM")).upper(), 0.70)
            for v in violations
        )
        weighted_severity_penalty = min(1.0, severity_sum / float(total_frames))

    burst_transitions = 0
    for idx in range(1, len(unique_frames)):
        if unique_frames[idx] - unique_frames[idx - 1] > 1:
            burst_transitions += 1
    temporal_penalty = min(0.35, burst_transitions * 0.04 + (0.08 if violated_count > 0 else 0.0))
    temporal_consistency = round(max(0.0, 1.0 - temporal_penalty - 0.35 * phr), 4)

    kinetic_penalty = min(0.20, max(0.0, kinetic_debt_joules) / 250.0)

    pir = round(
        max(
            0.0,
            min(
                1.0,
                1.0 - (0.50 * phr + 0.30 * weighted_severity_penalty + 0.10 * (1.0 - temporal_consistency) + kinetic_penalty),
            ),
        ),
        4,
    )

    rubric = classify_pir_rubric(pir)
    has_high_severity = any(str(v.get("severity", "")).upper() == "HIGH" for v in violations)
    verdict: VerdictType = "PASS" if (phr < 0.10 and pir >= 0.70 and not has_high_severity) else "FAIL"

    return {
        "physics_hallucination_rate": phr,
        "physical_integrity_rating": pir,
        "pir_grade": rubric["pir_grade"],
        "pir_label": rubric["pir_label"],
        "pir_meaning": rubric["pir_meaning"],
        "temporal_consistency_score": temporal_consistency,
        "violated_frames_count": violated_count,
        "overall_verdict": verdict,
    }
