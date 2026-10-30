"""Physics Auditor module (Module C) for Physical Fidelity Auditor.

Integrates Nebius Token Factory (NVIDIA Nemotron 3 Nano / Super) and Tavily API
to evaluate world-model predictions against Isaac Lab ground-truth simulations.
"""

from auditor.nemotron_client import (
    NEMOTRON_NANO_DEFAULT,
    NEMOTRON_SUPER_DEFAULT,
    TOKEN_FACTORY_DEFAULT_URL,
    NemotronPhysicsAuditor,
    analyze_kinematic_discrepancies,
    analyze_plan_steps_discrepancies,
)
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
from auditor.tavily_client import (
    EMPIRICAL_CONSTANTS_REFERENCE,
    PhysicalConstantsRetriever,
)

__all__ = [
    "AuditReport",
    "EMPIRICAL_CONSTANTS_REFERENCE",
    "NEMOTRON_NANO_DEFAULT",
    "NEMOTRON_SUPER_DEFAULT",
    "NemotronPhysicsAuditor",
    "PhysicalConstantsRetriever",
    "SYSTEM_PROMPT",
    "TOKEN_FACTORY_DEFAULT_URL",
    "ViolationItem",
    "analyze_kinematic_discrepancies",
    "analyze_plan_steps_discrepancies",
    "build_audit_prompt",
    "build_plan_audit_prompt",
    "build_super_escalation_prompt",
    "classify_pir_rubric",
    "compute_sipa_metrics",
]
