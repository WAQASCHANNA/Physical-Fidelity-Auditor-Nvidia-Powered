"""World Model Predictor module (Module B) for Physical Fidelity Auditor.

Generates action-conditioned future frame predictions ("dreams") using NVIDIA Cosmos Predict
(via `npa workbench cosmos`) or a controllable local world-model simulator that models
characteristic generative video physics hallucinations (contact interpenetration,
gravity drift, frictionless runaway, and object permanence flicker).
"""

from world_model.predictor import (
    HALLUCINATION_PROFILES,
    PredictionResult,
    generate_all_predictions,
    generate_world_model_prediction,
)

__all__ = [
    "HALLUCINATION_PROFILES",
    "PredictionResult",
    "generate_all_predictions",
    "generate_world_model_prediction",
]

