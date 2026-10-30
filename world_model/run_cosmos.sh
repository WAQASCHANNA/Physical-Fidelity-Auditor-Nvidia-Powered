#!/usr/bin/env bash
# Tier 2: World Model Prediction via Nebius Physical AI Workbench (Cosmos Predict)
# Falls back to the local Cosmos world-model predictor if `npa` CLI is not installed.

set -euo pipefail

SCENARIO="${1:-ramp_roll}"
GROUND_TRUTH_DIR="${2:-data/ground_truth}"
OUTPUT_DIR="${3:-data/predictions}"
PROFILE="${4:-cosmos_default}"

echo "[Tier 2: Cosmos Predict] Generating world-model dream for scenario='${SCENARIO}' (profile='${PROFILE}')..."

if command -v npa >/dev/null 2>&1 && [[ "${NPA_EXECUTION_MODE:-auto}" != "local" ]]; then
    echo "[Tier 2: Cosmos Predict] Provisioning Nebius Serverless Job via 'npa workbench cosmos'..."
    npa workbench cosmos predict \
        --input "${GROUND_TRUTH_DIR}/${SCENARIO}/frames/" \
        --output "${OUTPUT_DIR}/${SCENARIO}/"
else
    echo "[Tier 2: Cosmos Predict] Running local Cosmos world-model predictor..."
    python -m world_model.predictor \
        --scenario "${SCENARIO}" \
        --input "${GROUND_TRUTH_DIR}" \
        --output "${OUTPUT_DIR}" \
        --profile "${PROFILE}"
fi

echo "[Tier 2: Cosmos Predict] Completed scenario='${SCENARIO}' -> ${OUTPUT_DIR}/${SCENARIO}/"

