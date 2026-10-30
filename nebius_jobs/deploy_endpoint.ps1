<#
.SYNOPSIS
    Step 6: Containerize and Deploy Physical Fidelity Auditor to a Nebius Serverless Endpoint (PowerShell).
.EXAMPLE
    .\nebius_jobs\deploy_endpoint.ps1 -RegistryId "cr.eu-north1.nebius.cloud/<your-registry-id>"
#>
param(
    [string]$RegistryId = $env:NEBIUS_REGISTRY_ID,
    [string]$EndpointName = "physics-fidelity-auditor",
    [string]$ImageTag = "physics-auditor:v1"
)

$ErrorActionPreference = "Stop"

# Load keys from .env if not already in environment
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
            $k = $Matches[1].Trim()
            $v = $Matches[2].Trim()
            if (-not [Environment]::GetEnvironmentVariable($k)) {
                [Environment]::SetEnvironmentVariable($k, $v, "Process")
            }
        }
    }
}

$TokenFactoryKey = if ($env:NEBIUS_TOKEN_FACTORY_KEY) { $env:NEBIUS_TOKEN_FACTORY_KEY } else { $env:NEBIUS_API_KEY }
$TavilyKey = $env:TAVILY_API_KEY

if (-not $RegistryId) {
    $RegistryId = "<your-registry-id>"
}

Write-Host "[1/4] Authenticating Docker with Nebius Container Registry..." -ForegroundColor Cyan
nebius registry configure-docker

Write-Host "[2/4] Building and tagging Docker image: $RegistryId/$ImageTag..." -ForegroundColor Cyan
docker build -t $ImageTag .
docker tag $ImageTag "$RegistryId/$ImageTag"

Write-Host "[3/4] Pushing Docker image to Nebius Container Registry..." -ForegroundColor Cyan
docker push "$RegistryId/$ImageTag"

Write-Host "[4/4] Creating Nebius AI Serverless Endpoint: $EndpointName (port 8501)..." -ForegroundColor Cyan
nebius ai endpoint create `
  --name $EndpointName `
  --image "$RegistryId/$ImageTag" `
  --container-port 8501 `
  --platform cpu-d3 `
  --preset 4vcpu-16gb `
  --public `
  --env NEBIUS_TOKEN_FACTORY_KEY=$TokenFactoryKey `
  --env NEBIUS_API_KEY=$TokenFactoryKey `
  --env TAVILY_API_KEY=$TavilyKey

Write-Host "Deployment complete! Retrieve your public URL via: nebius ai endpoint get --name $EndpointName" -ForegroundColor Green

