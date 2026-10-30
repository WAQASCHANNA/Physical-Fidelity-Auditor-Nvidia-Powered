# Physical Fidelity Auditor (`physics-audit`)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Track: Physical AI](https://img.shields.io/badge/Track-Physical%20AI-76b900.svg)](#)
[![Nebius Token Factory](https://img.shields.io/badge/Nebius-Token%20Factory-00d4ff.svg)](https://api.tokenfactory.nebius.com/v1/)
[![NVIDIA Nemotron](https://img.shields.io/badge/NVIDIA-Nemotron%203%20Nano-76b900.svg)](#)
[![NVIDIA Cosmos](https://img.shields.io/badge/NVIDIA-Cosmos%203%20Reasoner-76b900.svg)](#)

> **License:** Distributed under the **MIT License** — see [`LICENSE`](LICENSE) at the root of this repository.  
> **Hackathon Track:** **Physical AI**

---

## 1. Project Description (What We Built, Why, and How It Works)

### Why We Built It (The Research Gap)
Generative world models frequently prioritize visual plausibility over dynamic physical consistency—producing **"physics hallucinations"** such as solid-surface interpenetration ($\text{SDF} < 0$), frictionless grasp slippage, ungrounded levitation against gravity, spontaneous kinetic energy creation, and object permanence loss. Existing benchmarks focus on visual quality (FVD, PSNR), leaving a critical gap in sim-to-real robotics: **measurement-based diagnostics that isolate violations of specific physical laws**.

### What We Built
**Physical Fidelity Auditor (`physics-audit`)** is a reproducible, automated 3-tier physical consistency benchmark and diagnostic studio deployed on **Nebius AI Cloud** and **Nebius Token Factory**:
1. **Tier 1 — 3D Ground Truth Oracle (`ground_truth/` — NVIDIA Isaac Lab):** Captures 3D robotic manipulation and rigid-body trajectories (`Isaac-Lift-Cube-Franka-v0`, `Isaac-Warehouse-AMR-Nav-v0`, `Isaac-Stack-Block-3D-v0`, `Isaac-Projectile-3D-v0`, `Isaac-Ramp-Roll-3D-v0`) with exact physical parameters (mass, static friction $\mu_s$, signed-distance clearance $\text{SDF}$, and minimum force-closure bounds).
2. **Tier 2 — 3D World Model Prediction (`npa/` & `world_model/` — NVIDIA Cosmos):** Feeds ground-truth RGB keyframes into **`nvidia/Cosmos3-Super-Reasoner`** on **Nebius Token Factory** (`npa workbench cosmos token-factory reason`) to generate predicted scene descriptions and multi-stage manipulation plans (`scene_reasoning.json`), alongside action-conditioned 3D rollouts.
3. **Tier 3 — Automated Physics Judge (`auditor/` — NVIDIA Nemotron + Tavily):** Queries **Tavily Search API** for empirical tribology/material friction constants and uses **`nvidia/Nemotron-3-Nano-30B-A3B`** on **Nebius Token Factory** (`response_format={"type": "json_object"}`) to audit every step of the Cosmos prediction against Isaac Lab ground truth, outputting the **Physics Hallucination Rate ($\text{PHR}$)** and **SIPA Physical Integrity Rating ($\text{PIR}$, Grades `A` / `C` / `F`)**.
4. **Tier 4 — Interactive 3D Simulation & Audit Studio (`dashboard/app.py` — Nebius Serverless Endpoint):** Full-screen NVIDIA Omniverse / Isaac Sim dark-mode 3D viewport with multi-camera sensors (`Lidar_01`, `Camera_Front`, `Camera_Rear`), Cosmos Dream hallucination toggles, and one-click live Nebius Token Factory + Tavily auditing.

---

## 2. System Architecture

```text
┌───────────────────────────────────────────────────────────────────────────┐
│                    TIER 1: 3D GROUND TRUTH GENERATION                     │
│  Nebius Serverless Job (npa workbench isaac-lab)                          │
│  - Tasks: Isaac-Lift-Cube-Franka-v0, Warehouse AMR, Block Stack, Ramp     │
│  - Outputs: data/ground_truth/franka_lift/frames/*.png + metadata.json    │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    TIER 2: WORLD MODEL SCENE REASONING                    │
│  Nebius Token Factory (nvidia/Cosmos3-Super-Reasoner)                     │
│  - CLI: npa workbench cosmos token-factory reason                         │
│  - Outputs: data/predictions/franka_lift/scene_reasoning.json             │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    TIER 3: NEMOTRON PHYSICS AUDITOR + TAVILY              │
│  Nebius Token Factory (nvidia/Nemotron-3-Nano-30B-A3B) + Tavily API       │
│  - Tavily retrieves empirical static friction & restitution ranges        │
│  - Nemotron audits contact_constraint, object_permanence, gravity_support,│
│    friction_slipping, and momentum_conservation                           │
│  - Outputs: data/reports/franka_lift_audit.json (PHR, PIR Grade A/C/F)    │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    TIER 4: STREAMLIT 3D SIMULATION STUDIO                 │
│  Nebius AI Serverless Endpoint (Docker Container on Port 8501)            │
│  - Real-time 3D viewport, LiDAR & multi-camera feeds, live audit trigger  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Nebius Stack & Partner Tool Usage (Explicit Callouts)

| Technology | Module / File | How It Is Used in `physics-audit` |
| :--- | :--- | :--- |
| **Nebius Token Factory** | [`auditor/nemotron_client.py`](auditor/nemotron_client.py), [`npa/workbench/token_factory.py`](npa/workbench/token_factory.py) | OpenAI-compatible serverless inference endpoint (`https://api.tokenfactory.nebius.com/v1/`) authenticated via `NEBIUS_TOKEN_FACTORY_KEY`. |
| **NVIDIA Nemotron** | [`auditor/nemotron_client.py`](auditor/nemotron_client.py), [`auditor/prompt_templates.py`](auditor/prompt_templates.py) | Uses **`nvidia/Nemotron-3-Nano-30B-A3B`** with `response_format={"type": "json_object"}` as the automated physics judge to audit Cosmos manipulation plans and trajectories, with automatic boundary escalation to **`nvidia/Nemotron-3-Super-120B-A12B`**. |
| **NVIDIA Cosmos** | [`npa/workbench/token_factory.py`](npa/workbench/token_factory.py), [`world_model/predictor.py`](world_model/predictor.py) | Uses **`nvidia/Cosmos3-Super-Reasoner`** (`npa workbench cosmos token-factory reason`) to ingest Isaac Lab RGB keyframes and output predicted 3D scene descriptions and manipulation plans (`scene_reasoning.json`). |
| **NVIDIA Isaac Lab** | [`ground_truth/capture_isaac_lab_scene_frames.py`](ground_truth/capture_isaac_lab_scene_frames.py), [`ground_truth/scenarios.py`](ground_truth/scenarios.py) | Generates 3D ground-truth frames and physical metadata (`metadata.json`, `trajectory.csv`) for `Isaac-Lift-Cube-Franka-v0` (`approach` → `pre-grasp` → `grasp` → `lift` → `hold`) and 4 additional 3D physics benchmarks. |
| **Tavily Search API** | [`auditor/tavily_client.py`](auditor/tavily_client.py) | Queries real-world engineering tribology sources for static friction ($\mu_s$) and restitution ($e$) constants (e.g., rubber gripper pads on ABS plastic cube, $\mu_s \in [0.55, 0.75]$) with disk caching in `data/cache/tavily_cache.json`. |
| **Nebius AI Cloud Serverless Endpoint & Jobs** | [`Dockerfile`](Dockerfile), [`nebius_jobs/deploy_endpoint.sh`](nebius_jobs/deploy_endpoint.sh), [`nebius_jobs/isaac-franka-capture-reason.yaml`](nebius_jobs/isaac-franka-capture-reason.yaml) | Packages the Streamlit 3D Simulation & Audit Studio into a container pushed via `nebius registry configure-docker` and deployed as a public HTTP service via `nebius ai endpoint create` on port `8501`. |

---

## 4. Setup & Step-by-Step Pipeline Execution

### Step 1: Clone & Configure Environment
```bash
# 1. Create and activate Python virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .\.venv\Scripts\Activate.ps1   # Windows PowerShell

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
export NEBIUS_TOKEN_FACTORY_KEY="v1.xxx"
export NEBIUS_API_KEY="$NEBIUS_TOKEN_FACTORY_KEY"
export TAVILY_API_KEY="tvly-xxx"
```

### Step 2: Generate 3D Ground-Truth Frames with Isaac Lab
```bash
# Path A: Quick validation with bundled sample frames (no GPU required)
python npa/examples/isaac_franka_token_factory_reason.py \
  --input-path docs/assets/hackathon/isaac-franka-lift-cube \
  --output-path data/isaac-franka-reason-out

# Path B: Capture 25 photorealistic 3D frames & metadata for Isaac-Lift-Cube-Franka-v0
python ground_truth/capture_isaac_lab_scene_frames.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --num-frames 25 \
  --output-dir data/ground_truth/franka_lift/frames
```
*Outputs:* `data/ground_truth/franka_lift/frames/frame_00.png` .. `frame_24.png` and `data/ground_truth/franka_lift/metadata.json`.

### Step 3: Generate Predicted "Dreams" with NVIDIA Cosmos Reasoner
```bash
python -m npa.cli workbench cosmos token-factory reason \
  --input-uri data/ground_truth/franka_lift/frames \
  --output-dir data/predictions/franka_lift \
  --model nvidia/Cosmos3-Super-Reasoner \
  --prompt "Analyze this sequence of a Franka robot arm attempting to lift a cube. Describe the physical interaction, object trajectories, and any potential anomalies."
```
*Outputs:* `data/predictions/franka_lift/scene_reasoning.json` and `data/predictions/franka_lift/metadata.json`.

### Step 4: Run the Nemotron Physics Auditor (+ Tavily Grounding)
```bash
python -m auditor.nemotron_client \
  --ground-truth data/ground_truth/franka_lift/metadata.json \
  --prediction data/predictions/franka_lift/scene_reasoning.json \
  --output data/reports/franka_lift_audit.json
```
*Outputs:* `data/reports/franka_lift_audit.json` containing `overall_verdict`, `physics_hallucination_rate` (`PHR`), `physical_integrity_rating` (`PIR`), `pir_grade` (`A` / `C` / `F`), `violations`, and `tavily_grounding`.

### Step 5: Launch the Interactive 3D Simulation & Audit Dashboard Locally
```bash
streamlit run dashboard/app.py --server.port=8501
```
Open `http://localhost:8501` in your browser.

### Step 6: Containerize and Deploy to a Nebius Serverless Endpoint
```bash
# 1. Authenticate Docker with Nebius Container Registry
nebius registry configure-docker

# 2. Build and tag the image
docker build -t physics-auditor:v1 .
docker tag physics-auditor:v1 <your-registry-id>/physics-auditor:v1

# 3. Push the image to Nebius Container Registry
docker push <your-registry-id>/physics-auditor:v1

# 4. Deploy as a public Nebius AI Serverless Endpoint on port 8501
nebius ai endpoint create \
  --name physics-fidelity-auditor \
  --image <your-registry-id>/physics-auditor:v1 \
  --container-port 8501 \
  --platform cpu-d3 \
  --preset 4vcpu-16gb \
  --public \
  --env NEBIUS_TOKEN_FACTORY_KEY=$NEBIUS_TOKEN_FACTORY_KEY \
  --env TAVILY_API_KEY=$TAVILY_API_KEY
```
Or run the automated deployment script:
```bash
# Bash
export NEBIUS_REGISTRY_ID="cr.eu-north1.nebius.cloud/<your-registry-id>"
bash nebius_jobs/deploy_endpoint.sh

# Windows PowerShell
.\nebius_jobs\deploy_endpoint.ps1 -RegistryId "cr.eu-north1.nebius.cloud/<your-registry-id>"
```

### Run Unit & Integration Tests
```bash
pytest -v
```

---

## 5. Evaluation Metrics & SIPA Physical Integrity Rubric

* **Physics Hallucination Rate ($\text{PHR}$):**
  $$\text{PHR} = \frac{\text{Violated Steps (or Frames)}}{\text{Total Steps (or Frames)}} \in [0.0, 1.0]$$
* **SIPA Physical Integrity Rating ($\text{PIR}$):**

| PIR Score Range | Grade | Label | Physical Meaning |
| :--- | :---: | :--- | :--- |
| **`0.70 – 1.00`** | **`A`** | `Physically Solvent` | Model respects conservation laws and contact mechanics. |
| **`0.30 – 0.69`** | **`C`** | `Moderate Risk` | Some physical debt; may fail in novel situations. |
| **`0.00 – 0.29`** | **`F`** | `Critical` | Physical bankruptcy; trajectory violates fundamental causality. |

---

## 6. Devpost Submission Checklist & Written Feedback

### 6.1 Devpost Submission Verification
| Requirement | Status | Reference |
| :--- | :--- | :--- |
| **Track selected** | ✅ **Physical AI** | Top of `README.md` |
| **Public code repository with MIT License at top** | ✅ **MIT License** | [`LICENSE`](LICENSE) & badge at top of `README.md` |
| **README with setup & explicit callouts** | ✅ **Complete** | Sections 3 & 4 call out **Nemotron**, **Token Factory**, **Cosmos**, **Isaac Lab**, and **Tavily** |
| **Dockerfile & Nebius Serverless Endpoint deployment** | ✅ **Complete** | [`Dockerfile`](Dockerfile), [`nebius_jobs/deploy_endpoint.sh`](nebius_jobs/deploy_endpoint.sh), [`nebius_jobs/deploy_endpoint.ps1`](nebius_jobs/deploy_endpoint.ps1) |
| **Demo Video (≤ 3 min)** | 🎬 **Ready to record** | Walkthrough script below |

### 6.2 3-Minute Demo Video Script (`Isaac Lab → Cosmos → Nemotron → Dashboard`)
* **0:00 – 0:35 | The Problem:** Generative world models look photorealistic on screen but hallucinate physics—interpenetrating tables, dropping grasp force to $0\text{ N}$ while lifting objects, or losing object permanence.
* **0:35 – 1:15 | Tier 1 (Isaac Lab) & Tier 2 (Cosmos on Token Factory):** Show `ground_truth/capture_isaac_lab_scene_frames.py` generating the 25 3D keyframes and `metadata.json` for `Isaac-Lift-Cube-Franka-v0`, followed by `npa workbench cosmos token-factory reason` producing `data/predictions/franka_lift/scene_reasoning.json`.
* **1:15 – 2:00 | Tier 3 (Nemotron Physics Auditor + Tavily):** Run `python -m auditor.nemotron_client` live. Highlight Tavily retrieving real-world friction constants ($\mu = 0.65$ for rubber on ABS) and `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` on Nebius Token Factory outputting `data/reports/franka_lift_audit.json` (`overall_verdict: FAIL`, `PHR: 0.80`, `PIR: 0.10` Grade `F`).
* **2:00 – 3:00 | Tier 4 (3D Simulation Studio Dashboard on Nebius Serverless Endpoint):** Open the Streamlit 3D Simulation Studio (`http://localhost:8501` / Nebius Endpoint URL). Scrub the timeline, toggle **Cosmos Dream (Hallucinate)** vs. **Isaac Lab GT**, switch between **Warehouse AMR**, **Franka Lift Cube**, **Block Stack**, **Projectile**, and **Ramp Roll**, and click **Re-Run Step 4 Plan Audit** in the bottom drawer.

### 6.3 Written Feedback on Nebius Token Factory, AI Cloud, and NVIDIA Tools
* **Nebius Token Factory (`https://api.tokenfactory.nebius.com/v1/`):** Drop-in OpenAI SDK compatibility made wiring both **`nvidia/Nemotron-3-Nano-30B-A3B`** and **`nvidia/Cosmos3-Super-Reasoner`** effortless. Structured JSON output (`response_format={"type": "json_object"}`) worked reliably with Nemotron's `<think>` reasoning traces, enabling deterministic schema validation with Pydantic.
* **Nebius Physical AI Workbench (`npa`) & Serverless Endpoints:** Separating GPU-bound simulation/capture jobs (`isaac-lab` on L40S) from lightweight CPU-hosted reasoning and Streamlit visualization (`nebius ai endpoint create --platform cpu-d3 --preset 4vcpu-16gb`) drastically reduced iteration cost while keeping the interactive dashboard responsive.
* **NVIDIA Isaac Lab + Cosmos + Nemotron Synergy:** Pairing Isaac Lab's exact ground-truth state metadata with Cosmos 3's multi-stage manipulation planning and Nemotron 3 Nano's step-by-step physical constraint auditing creates a closed-loop evaluator that pinpoints *why* a world model fails before deploying policies to physical robots.
