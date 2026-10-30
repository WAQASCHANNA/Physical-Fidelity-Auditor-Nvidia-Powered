#!/usr/bin/env bash
# Tier 1: Ground Truth Generation via Nebius Physical AI Workbench (Isaac Lab)
# Falls back to the local analytical/symplectic Isaac Lab simulator if `npa` CLI is not present.

set -euo pipefail

SCENARIO="${1:-ramp_roll}"
OUTPUT_DIR="${2:-data/ground_truth}"
NUM_FRAMES="${3:-101}"

echo "[Tier 1: Isaac Lab] Starting ground-truth generation for scenario='${SCENARIO}'..."

if command -v npa >/dev/null 2>&1 && [[ "${NPA_EXECUTION_MODE:-auto}" != "local" ]]; then
    echo "[Tier 1: Isaac Lab] Provisioning Nebius Serverless Job via 'npa workbench isaac-lab'..."
    npa workbench isaac-lab run \
        --scenario "${SCENARIO}" \
        --num-frames "${NUM_FRAMES}" \
        --output "${OUTPUT_DIR}/${SCENARIO}/"
else
    echo "[Tier 1: Isaac Lab] Using local deterministic physics engine fallback..."
    python -m ground_truth.scenarios \
        --scenario "${SCENARIO}" \
        --output "${OUTPUT_DIR}" \
        --num-frames "${NUM_FRAMES}"
fi

echo "[Tier 1: Isaac Lab] Completed scenario='${SCENARIO}' -> ${OUTPUT_DIR}/${SCENARIO}/"

