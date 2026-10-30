#!/usr/bin/env bash
# =============================================================================
# Step 6: Containerize and Deploy Physical Fidelity Auditor to Nebius Serverless Endpoint
# Usage:
#   export NEBIUS_REGISTRY_ID="cr.eu-north1.nebius.cloud/<your-registry-id>"
#   export NEBIUS_TOKEN_FACTORY_KEY="v1.xxx"
#   export TAVILY_API_KEY="tvly-xxx"
#   bash nebius_jobs/deploy_endpoint.sh
# =============================================================================
set -euo pipefail

REGISTRY_ID="${NEBIUS_REGISTRY_ID:-<your-registry-id>}"
IMAGE_NAME="physics-auditor:v1"
ENDPOINT_NAME="${NEBIUS_ENDPOINT_NAME:-physics-fidelity-auditor}"

if [[ -z "${NEBIUS_TOKEN_FACTORY_KEY:-}" && -f ".env" ]]; then
  # shellcheck disable=SC1091
  set -a && source .env && set +a
  export NEBIUS_TOKEN_FACTORY_KEY="${NEBIUS_TOKEN_FACTORY_KEY:-${NEBIUS_API_KEY:-}}"
fi

echo "[1/4] Authenticating Docker with Nebius Container Registry..."
nebius registry configure-docker

echo "[2/4] Building and tagging Docker image: ${REGISTRY_ID}/${IMAGE_NAME}..."
docker build -t "${IMAGE_NAME}" .
docker tag "${IMAGE_NAME}" "${REGISTRY_ID}/${IMAGE_NAME}"

echo "[3/4] Pushing Docker image to Nebius Container Registry..."
docker push "${REGISTRY_ID}/${IMAGE_NAME}"

echo "[4/4] Creating Nebius AI Serverless Endpoint: ${ENDPOINT_NAME} (port 8501)..."
nebius ai endpoint create \
  --name "${ENDPOINT_NAME}" \
  --image "${REGISTRY_ID}/${IMAGE_NAME}" \
  --container-port 8501 \
  --platform cpu-d3 \
  --preset 4vcpu-16gb \
  --public \
  --env NEBIUS_TOKEN_FACTORY_KEY="${NEBIUS_TOKEN_FACTORY_KEY}" \
  --env NEBIUS_API_KEY="${NEBIUS_TOKEN_FACTORY_KEY}" \
  --env TAVILY_API_KEY="${TAVILY_API_KEY}"

echo "Deployment complete! Run 'nebius ai endpoint get --name ${ENDPOINT_NAME}' to view your public https:// URL."

