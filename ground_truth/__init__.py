"""Ground Truth Generator module (Module A) for Physical Fidelity Auditor.

Generates physically unambiguous rigid-body trajectories and rendered frame sequences
using NVIDIA Isaac Lab (via `npa workbench isaac-lab`) or a deterministic local
rigid-body physics engine with exact analytical/symplectic integration.
"""

from ground_truth.scenarios import (
    SCENARIO_REGISTRY,
    ScenarioConfig,
    SimulationResult,
    generate_all_scenarios,
    generate_ground_truth,
    get_scenario_config,
)

__all__ = [
    "SCENARIO_REGISTRY",
    "ScenarioConfig",
    "SimulationResult",
    "generate_all_scenarios",
    "generate_ground_truth",
    "get_scenario_config",
]

