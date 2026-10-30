"""Module D: NVIDIA Omniverse / Isaac Sim 3D Environment & Physics Auditor Dashboard (dashboard/app.py).

Renders the full-screen dark-mode 3D Robotics & Warehouse Simulation Studio UI:
- Top Bar: Simulation | Scene | Navigation | Sensors | Behavior | Diagnostics + Physics Audit Mode
- Left Dock: Expandable USD Scene Hierarchy Tree (Warehouse, Environment, Racks, Robots, Objects, Sensors)
- Center Viewport: Real-time 3D RTX-style Warehouse & Robotics Viewport (Robot_01 AMR, 360° Lidar_01 rays,
  Racks A4/A5/B4/B5, Forklift_01 with 3D bounding boxes, green/purple navigation spline, and Cosmos Physics Hallucination overlays)
- Timeline Scrubber: Play/Pause, 00:01:23 / 00:05:00 timecode, speed selector, and violation markers
- Bottom Dock: Console/Assets/Tasks log, Live Multi-Sensor strip (Lidar_01, Camera_Front, Camera_Rear), and GPU/CPU/FPS/PHR Metrics
- Right Dock: Robot 01 Inspector (Transform X/Y/Z, Status Battery/Speed/Mode, Navigation + 2D Top-Down Occupancy Minimap, Simulation Time)
- Integrated Nebius Token Factory (Nemotron 3 Nano) + Tavily Physics Audit drawer
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import ground_truth.scenarios as _gt_scenarios
import world_model.predictor as _wm_predictor
import pipeline as _pipeline

importlib.reload(_gt_scenarios)
importlib.reload(_wm_predictor)
importlib.reload(_pipeline)

from ground_truth.scenarios import SCENARIO_REGISTRY
from pipeline import run_full_benchmark, run_scenario_pipeline
from world_model.predictor import HALLUCINATION_PROFILES


DATA_DIR = REPO_ROOT / "data"


def _ensure_reports_exist() -> None:
    reports_dir = DATA_DIR / "reports"
    if not (reports_dir / "warehouse_amr_audit.json").exists():
        run_scenario_pipeline(scenario="warehouse_amr", profile="cosmos_default", data_dir=DATA_DIR)
    if not (reports_dir / "franka_lift_cube_audit.json").exists():
        run_scenario_pipeline(scenario="franka_lift_cube", profile="cosmos_default", data_dir=DATA_DIR)


def _load_all_reports() -> Dict[str, Dict[str, Any]]:
    _ensure_reports_exist()
    reports: Dict[str, Dict[str, Any]] = {}
    for scen in SCENARIO_REGISTRY:
        p = DATA_DIR / "reports" / f"{scen}_audit.json"
        if p.exists():
            reports[scen] = json.loads(p.read_text(encoding="utf-8"))
    return reports


def _build_simulation_studio_html(reports_json_str: str) -> str:
    """Return the self-contained HTML5/Canvas/WebGL 3D Simulation Environment Dashboard."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<style>
  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    user-select: none;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }}
  body {{
    background: #0e1015;
    color: #d1d5db;
    height: 100vh;
    width: 100vw;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    font-size: 12px;
  }}

  /* TOP NAVIGATION BAR */
  .topbar {{
    height: 38px;
    background: #14171f;
    border-bottom: 1px solid #262b36;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 14px;
    flex-shrink: 0;
  }}
  .topbar-left {{
    display: flex;
    align-items: center;
    gap: 22px;
  }}
  .brand {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
    font-size: 13px;
    color: #f3f4f6;
    letter-spacing: 0.2px;
  }}
  .brand-cube {{
    width: 18px;
    height: 18px;
    background: linear-gradient(135deg, #4ade80 0%, #16a34a 100%);
    clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
    display: inline-block;
    box-shadow: 0 0 10px rgba(74, 222, 128, 0.45);
  }}
  .nav-tabs {{
    display: flex;
    gap: 18px;
    color: #9ca3af;
    font-size: 12px;
  }}
  .nav-tab {{
    cursor: pointer;
    padding: 8px 2px;
    border-bottom: 2px solid transparent;
    transition: color 0.15s, border-color 0.15s;
  }}
  .nav-tab:hover {{ color: #e5e7eb; }}
  .nav-tab.active {{
    color: #ffffff;
    border-bottom: 2px solid #4ade80;
    font-weight: 500;
  }}
  .topbar-right {{
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .audit-pill {{
    background: #1e2430;
    border: 1px solid #334155;
    border-radius: 5px;
    padding: 4px 10px;
    font-size: 11px;
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
  }}
  .audit-pill.hallucination-active {{
    background: rgba(153, 27, 27, 0.35);
    border-color: #ef4444;
    color: #fca5a5;
  }}
  .user-icon {{
    width: 24px;
    height: 24px;
    border-radius: 50%;
    border: 1px solid #4b5563;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #9ca3af;
    font-size: 12px;
  }}

  /* MAIN WORKSPACE GRID */
  .workspace {{
    flex: 1;
    display: grid;
    grid-template-columns: 225px 1fr 270px;
    grid-template-rows: 1fr 195px;
    gap: 4px;
    padding: 4px;
    background: #0b0d11;
    min-height: 0;
  }}

  .panel {{
    background: #161920;
    border: 1px solid #242936;
    border-radius: 4px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}
  .panel-header {{
    height: 28px;
    padding: 0 10px;
    background: #1a1e27;
    border-bottom: 1px solid #242936;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-weight: 600;
    color: #e5e7eb;
    font-size: 11.5px;
    flex-shrink: 0;
  }}

  /* LEFT SCENE TREE PANEL (spans both rows on column 1? In screenshot, Scene is top-left and Console is bottom-left) */
  .scene-panel {{
    grid-column: 1 / 2;
    grid-row: 1 / 2;
  }}
  .search-box {{
    margin: 6px 8px;
    background: #101319;
    border: 1px solid #282e3d;
    border-radius: 4px;
    padding: 4px 8px;
    display: flex;
    align-items: center;
    gap: 6px;
    color: #6b7280;
  }}
  .search-box input {{
    background: transparent;
    border: none;
    outline: none;
    color: #d1d5db;
    font-size: 11px;
    width: 100%;
  }}
  .tree-container {{
    flex: 1;
    overflow-y: auto;
    padding: 2px 4px 8px 4px;
  }}
  .tree-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 3.5px 6px;
    border-radius: 3px;
    cursor: pointer;
    color: #cbd5e1;
    font-size: 11.5px;
  }}
  .tree-row:hover {{ background: #202531; }}
  .tree-row.selected {{
    background: #283040;
    color: #ffffff;
    font-weight: 500;
  }}
  .tree-left {{
    display: flex;
    align-items: center;
    gap: 6px;
    overflow: hidden;
    white-space: nowrap;
  }}
  .eye-btn {{
    color: #64748b;
    font-size: 11px;
    cursor: pointer;
    padding: 0 2px;
  }}
  .eye-btn.hidden-node {{
    opacity: 0.3;
  }}

  /* CENTER 3D VIEWPORT + TIMELINE */
  .viewport-panel {{
    grid-column: 2 / 3;
    grid-row: 1 / 2;
    position: relative;
    background: #0d1017;
  }}
  .vp-Con-wrap {{
    position: relative;
    flex: 1;
    width: 100%;
    height: 100%;
    overflow: hidden;
  }}
  #main3dCanvas {{
    width: 100%;
    height: 100%;
    display: block;
    cursor: grab;
  }}
  #main3dCanvas:active {{ cursor: grabbing; }}

  /* Viewport Floating Top Overlay */
  .vp-overlay-top {{
    position: absolute;
    top: 8px;
    left: 10px;
    right: 10px;
    display: flex;
    justify-content: space-between;
    pointer-events: none;
  }}
  .vp-toolbar {{
    display: flex;
    align-items: center;
    gap: 4px;
    background: rgba(20, 23, 31, 0.88);
    border: 1px solid #2e3646;
    border-radius: 4px;
    padding: 3px 6px;
    pointer-events: auto;
    backdrop-filter: blur(6px);
  }}
  .vp-btn {{
    padding: 3px 7px;
    border-radius: 3px;
    color: #d1d5db;
    font-size: 11px;
    cursor: pointer;
    background: transparent;
    border: none;
  }}
  .vp-btn:hover, .vp-btn.active {{
    background: #2b3344;
    color: #fff;
  }}
  .vp-select {{
    background: #181c26;
    color: #e5e7eb;
    border: 1px solid #323b4e;
    border-radius: 3px;
    padding: 2px 6px;
    font-size: 11px;
    outline: none;
    cursor: pointer;
  }}
  .rtx-badge {{
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    color: #e5e7eb;
    padding: 2px 8px;
  }}
  .rtx-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 6px #4ade80;
  }}

  /* TIMELINE SCRUBBER BAR */
  .timeline-bar {{
    height: 36px;
    background: #151821;
    border-top: 1px solid #242936;
    display: flex;
    align-items: center;
    padding: 0 12px;
    gap: 12px;
    flex-shrink: 0;
  }}
  .tl-label {{
    color: #9ca3af;
    font-size: 11px;
    font-weight: 500;
  }}
  .tl-controls {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .tl-btn {{
    background: #1f2430;
    border: 1px solid #2e3646;
    color: #e5e7eb;
    border-radius: 3px;
    width: 22px;
    height: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 10px;
  }}
  .tl-btn:hover {{ background: #2d3548; }}
  .tl-time {{
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 11px;
    color: #cbd5e1;
    min-width: 112px;
  }}
  .tl-slider-wrap {{
    flex: 1;
    position: relative;
    display: flex;
    align-items: center;
  }}
  #timelineSlider {{
    width: 100%;
    accent-color: #3b82f6;
    cursor: pointer;
    height: 4px;
  }}

  /* RIGHT INSPECTOR PANEL (spans rows 1 & 2 on column 3) */
  .inspector-panel {{
    grid-column: 3 / 4;
    grid-row: 1 / 3;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }}
  .status-active {{
    color: #4ade80;
    font-size: 11px;
    font-weight: 500;
  }}
  .insp-tabs {{
    display: flex;
    border-bottom: 1px solid #242936;
    background: #151820;
    padding: 0 8px;
  }}
  .insp-tab {{
    padding: 6px 10px;
    font-size: 11px;
    color: #9ca3af;
    cursor: pointer;
    border-bottom: 2px solid transparent;
  }}
  .insp-tab.active {{
    color: #3b82f6;
    border-bottom: 2px solid #3b82f6;
    font-weight: 500;
  }}
  .insp-body {{
    flex: 1;
    overflow-y: auto;
    padding: 8px 10px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }}
  .insp-section {{
    background: #1b1f2a;
    border: 1px solid #262c3a;
    border-radius: 4px;
    padding: 8px;
  }}
  .sec-title {{
    font-size: 11px;
    font-weight: 600;
    color: #e5e7eb;
    margin-bottom: 7px;
    display: flex;
    justify-content: space-between;
  }}
  .prop-label {{
    color: #9ca3af;
    font-size: 10.5px;
    margin-bottom: 3px;
  }}
  .xyz-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 5px;
    margin-bottom: 6px;
  }}
  .xyz-box {{
    background: #12151c;
    border: 1px solid #262c3a;
    border-radius: 3px;
    padding: 3px 5px;
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 10.5px;
  }}
  .xyz-axis-x {{ color: #f87171; font-weight: 600; }}
  .xyz-axis-y {{ color: #4ade80; font-weight: 600; }}
  .xyz-axis-z {{ color: #60a5fa; font-weight: 600; }}

  .kv-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 3px 0;
    font-size: 11px;
  }}
  .kv-key {{ color: #9ca3af; }}
  .kv-val {{ color: #f3f4f6; font-weight: 500; font-family: 'JetBrains Mono', Consolas, monospace; }}
  .progress-track {{
    height: 5px;
    background: #10131a;
    border-radius: 3px;
    overflow: hidden;
    margin: 3px 0 6px 0;
  }}
  .progress-fill-green {{
    height: 100%;
    background: linear-gradient(90deg, #22c55e, #4ade80);
  }}
  .progress-fill-blue {{
    height: 100%;
    background: linear-gradient(90deg, #2563eb, #60a5fa);
  }}
  #minimapCanvas {{
    width: 100%;
    height: 105px;
    background: #0f1218;
    border: 1px solid #283040;
    border-radius: 3px;
    margin-top: 4px;
    display: block;
  }}

  /* BOTTOM DOCK ROW */
  .console-panel {{
    grid-column: 1 / 2;
    grid-row: 2 / 3;
  }}
  .console-logs {{
    flex: 1;
    padding: 6px 8px;
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 10.5px;
    line-height: 1.55;
    overflow-y: auto;
    background: #12151c;
  }}
  .log-time {{ color: #64748b; }}
  .log-info {{ color: #38bdf8; }}
  .log-warn {{ color: #facc15; }}
  .log-err {{ color: #f87171; font-weight: 600; }}

  /* BOTTOM CENTER SENSORS + METRICS WRAPPER */
  .bottom-center-grid {{
    grid-column: 2 / 3;
    grid-row: 2 / 3;
    display: grid;
    grid-template-columns: 1fr 225px;
    gap: 4px;
    min-height: 0;
  }}
  .sensors-strip {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
    padding: 6px 8px;
    flex: 1;
    min-height: 0;
  }}
  .sensor-card {{
    background: #10131a;
    border: 1px solid #242936;
    border-radius: 3px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}
  .sensor-title {{
    font-size: 10.5px;
    color: #9ca3af;
    padding: 3px 6px;
    background: #161a23;
    border-bottom: 1px solid #222733;
    display: flex;
    justify-content: space-between;
  }}
  .sensor-canvas {{
    flex: 1;
    width: 100%;
    height: 100%;
    display: block;
  }}
  .metrics-body {{
    padding: 8px 10px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    flex: 1;
  }}
</style>
</head>
<body>

  <!-- TOP NAVIGATION BAR -->
  <div class="topbar">
    <div class="topbar-left">
      <div class="brand">
        <span class="brand-cube"></span>
        <span>Simulation</span>
      </div>
      <div class="nav-tabs">
        <span class="nav-tab active" onclick="switchTopTab(this, 'Scene')">Scene</span>
        <span class="nav-tab" onclick="switchTopTab(this, 'Navigation')">Navigation</span>
        <span class="nav-tab" onclick="switchTopTab(this, 'Sensors')">Sensors</span>
        <span class="nav-tab" onclick="switchTopTab(this, 'Behavior')">Behavior</span>
        <span class="nav-tab" onclick="switchTopTab(this, 'Diagnostics')">Diagnostics</span>
      </div>
    </div>

    <div class="topbar-right">
      <select class="vp-select" id="scenarioSelect" onchange="changeScenario(this.value)" title="Select 3D Physical Scenario">
        <option value="warehouse_amr">Scene: Warehouse AMR (Robot 01)</option>
        <option value="franka_lift_cube">Scene: Franka Lift Cube (Isaac-Lift-Cube-Franka-v0)</option>
        <option value="block_stack">Scene: 3D Block Stack & Topple</option>
        <option value="projectile">Scene: 3D Projectile Motion</option>
      </select>
      <div class="audit-pill" id="cosmosModeBtn" onclick="toggleCosmosDreamMode()" title="Toggle between Isaac Lab Ground Truth and Cosmos World Model Dream">
        <span id="cosmosModeDot" style="color:#4ade80;">●</span>
        <span id="cosmosModeLabel">Mode: Isaac Lab 3D Ground Truth</span>
      </div>
      <div class="audit-pill" id="phrScoreBadge" title="Nemotron 3 Nano + Tavily SIPA Audit Score">
        <span>⚖️ PHR: <strong id="phrVal" style="color:#4ade80;">0.000</strong></span>
        <span>|</span>
        <span>PIR: <strong id="pirVal" style="color:#38bdf8;">0.985</strong></span>
      </div>
      <div class="user-icon">👤</div>
    </div>
  </div>

  <!-- MAIN WORKSPACE -->
  <div class="workspace">

    <!-- LEFT PANEL: SCENE TREE -->
    <div class="panel scene-panel">
      <div class="panel-header">
        <span>Scene</span>
        <span style="color:#6b7280;cursor:pointer;">✕</span>
      </div>
      <div class="search-box">
        <span>🔍</span>
        <input type="text" id="treeSearch" placeholder="Search" oninput="filterSceneTree(this.value)" />
      </div>
      <div class="tree-container" id="sceneTree">
        <div class="tree-row" data-name="warehouse"><div class="tree-left"><span>▾</span><span>📁</span><span>Warehouse</span></div><span class="eye-btn" onclick="toggleLayer('warehouse', this)">👁</span></div>
        <div class="tree-row" data-name="environment" style="padding-left:18px;"><div class="tree-left"><span>▾</span><span>📁</span><span>Environment</span></div><span class="eye-btn" onclick="toggleLayer('environment', this)">👁</span></div>
        <div class="tree-row" data-name="floor" style="padding-left:34px;"><div class="tree-left"><span>🧊</span><span>Floor</span></div><span class="eye-btn" onclick="toggleLayer('floor', this)">👁</span></div>
        <div class="tree-row" data-name="lighting" style="padding-left:34px;"><div class="tree-left"><span>☀️</span><span>Lighting</span></div><span class="eye-btn" onclick="toggleLayer('lighting', this)">👁</span></div>
        <div class="tree-row" data-name="walls" style="padding-left:34px;"><div class="tree-left"><span>🧱</span><span>Walls</span></div><span class="eye-btn" onclick="toggleLayer('walls', this)">👁</span></div>

        <div class="tree-row" data-name="racks" style="padding-left:18px;"><div class="tree-left"><span>▾</span><span>📁</span><span>Racks</span></div><span class="eye-btn" onclick="toggleLayer('racks', this)">👁</span></div>
        <div class="tree-row" data-name="rack_a" style="padding-left:34px;" onclick="selectEntity('Rack_A', this)"><div class="tree-left"><span>🗄️</span><span>Rack_A</span></div><span class="eye-btn" onclick="event.stopPropagation();toggleLayer('racks', this)">👁</span></div>
        <div class="tree-row" data-name="rack_b" style="padding-left:34px;" onclick="selectEntity('Rack_B', this)"><div class="tree-left"><span>🗄️</span><span>Rack_B</span></div><span class="eye-btn" onclick="event.stopPropagation();toggleLayer('racks', this)">👁</span></div>
        <div class="tree-row" data-name="rack_c" style="padding-left:34px;" onclick="selectEntity('Rack_C', this)"><div class="tree-left"><span>🗄️</span><span>Rack_C</span></div><span class="eye-btn" onclick="event.stopPropagation();toggleLayer('racks', this)">👁</span></div>

        <div class="tree-row" data-name="robots" style="padding-left:18px;"><div class="tree-left"><span>▾</span><span>📁</span><span>Robots</span></div><span class="eye-btn" onclick="toggleLayer('robots', this)">👁</span></div>
        <div class="tree-row selected" data-name="robot_01" style="padding-left:34px;" onclick="selectEntity('Robot 01', this)"><div class="tree-left"><span>🤖</span><span>Robot_01</span></div><span class="eye-btn" onclick="event.stopPropagation();toggleLayer('robot_01', this)">👁</span></div>
        <div class="tree-row" data-name="forklift_01" style="padding-left:34px;" onclick="selectEntity('Forklift 01', this)"><div class="tree-left"><span>🚜</span><span>Forklift_01</span></div><span class="eye-btn" onclick="event.stopPropagation();toggleLayer('forklift_01', this)">👁</span></div>

        <div class="tree-row" data-name="objects" style="padding-left:18px;"><div class="tree-left"><span>▾</span><span>📁</span><span>Objects</span></div><span class="eye-btn" onclick="toggleLayer('objects', this)">👁</span></div>
        <div class="tree-row" data-name="pallet_01" style="padding-left:34px;"><div class="tree-left"><span>📦</span><span>Pallet_01</span></div><span class="eye-btn" onclick="toggleLayer('objects', this)">👁</span></div>
        <div class="tree-row" data-name="pallet_02" style="padding-left:34px;"><div class="tree-left"><span>📦</span><span>Pallet_02</span></div><span class="eye-btn" onclick="toggleLayer('objects', this)">👁</span></div>
        <div class="tree-row" data-name="box_01" style="padding-left:34px;"><div class="tree-left"><span>📦</span><span>Box_01</span></div><span class="eye-btn" onclick="toggleLayer('objects', this)">👁</span></div>
        <div class="tree-row" data-name="box_02" style="padding-left:34px;"><div class="tree-left"><span>📦</span><span>Box_02</span></div><span class="eye-btn" onclick="toggleLayer('objects', this)">👁</span></div>
        <div class="tree-row" data-name="box_03" style="padding-left:34px;"><div class="tree-left"><span>📦</span><span>Box_03</span></div><span class="eye-btn" onclick="toggleLayer('objects', this)">👁</span></div>

        <div class="tree-row" data-name="sensors" style="padding-left:18px;"><div class="tree-left"><span>▾</span><span>📁</span><span>Sensors</span></div><span class="eye-btn" onclick="toggleLayer('lidar', this)">👁</span></div>
        <div class="tree-row" data-name="lidar_01" style="padding-left:34px;"><div class="tree-left"><span>📡</span><span>Lidar_01</span></div><span class="eye-btn" onclick="toggleLayer('lidar', this)">👁</span></div>
        <div class="tree-row" data-name="camera_front" style="padding-left:34px;"><div class="tree-left"><span>📷</span><span>Camera_Front</span></div><span class="eye-btn" onclick="toggleLayer('cameras', this)">👁</span></div>
        <div class="tree-row" data-name="camera_rear" style="padding-left:34px;"><div class="tree-left"><span>📷</span><span>Camera_Rear</span></div><span class="eye-btn" onclick="toggleLayer('cameras', this)">👁</span></div>
      </div>
    </div>

    <!-- CENTER 3D VIEWPORT + TIMELINE -->
    <div class="panel viewport-panel">
      <div class="vp-Con-wrap">
        <canvas id="main3dCanvas"></canvas>

        <!-- Viewport Overlay Toolbars -->
        <div class="vp-overlay-top">
          <div class="vp-toolbar">
            <button class="vp-btn active">3D ▾</button>
            <button class="vp-btn" title="Translate Gizmo" onclick="resetCamera()">✥</button>
            <button class="vp-btn" title="Rotate Orbit">↻</button>
            <button class="vp-btn" title="Bounding Boxes" onclick="toggleBBoxes()">⚃</button>
            <select class="vp-select" id="cameraViewSelect" onchange="setCameraPreset(this.value)">
              <option value="perspective">Perspective</option>
              <option value="chase">Robot_01 Chase</option>
              <option value="top">Top-Down Ortho</option>
              <option value="isometric">Isometric Audit</option>
            </select>
          </div>

          <div class="vp-toolbar">
            <button class="vp-btn" onclick="toggleLayer('lidar')" title="Toggle LiDAR Rays">📡</button>
            <button class="vp-btn" onclick="toggleCosmosDreamMode()" title="Toggle Physics Audit Overlay">⚖️</button>
            <div class="rtx-badge">
              <span class="rtx-dot" id="rtxDot"></span>
              <span id="rtxText">RTX - Real Time</span>
            </div>
            <button class="vp-btn" id="vpPlayPauseBtn" onclick="togglePlay()">⏸</button>
          </div>
        </div>
      </div>

      <!-- Timeline Bar -->
      <div class="timeline-bar">
        <span class="tl-label">Timeline</span>
        <div class="tl-controls">
          <button class="tl-btn" onclick="restartSim()" title="Restart">⏮</button>
          <button class="tl-btn" id="tlPlayBtn" onclick="togglePlay()" title="Play/Pause">⏸</button>
          <button class="tl-btn" onclick="stepFrame()" title="Step Frame">⏭</button>
        </div>
        <span class="tl-time" id="timecodeDisplay">00:01:23 / 00:05:00</span>
        <span class="tl-label" style="color:#e5e7eb;">1.0x</span>
        <div class="tl-slider-wrap">
          <input type="range" id="timelineSlider" min="0" max="240" value="83" oninput="scrubTimeline(this.value)" />
        </div>
        <span style="color:#64748b;font-size:10.5px;font-family:monospace;" id="frameCounterLabel">Frame 083 / 240</span>
      </div>
    </div>

    <!-- RIGHT PANEL: ROBOT 01 INSPECTOR -->
    <div class="panel inspector-panel">
      <div>
        <div class="panel-header">
          <span id="inspectorEntityTitle">Robot 01</span>
          <span class="status-active" id="inspectorEntityBadge">Active</span>
        </div>
        <div class="insp-tabs">
          <div class="insp-tab active" onclick="switchInspTab(this)">Overview</div>
          <div class="insp-tab" onclick="switchInspTab(this)">Sensors</div>
          <div class="insp-tab" onclick="switchInspTab(this)">Behavior</div>
        </div>
      </div>

      <div class="insp-body">
        <!-- Transform Section -->
        <div class="insp-section">
          <div class="sec-title"><span>Transform</span></div>
          <div class="prop-label">Position</div>
          <div class="xyz-grid">
            <div class="xyz-box"><span class="xyz-axis-x">X</span><span id="posX">12.42</span></div>
            <div class="xyz-box"><span class="xyz-axis-y">Y</span><span id="posY">4.85</span></div>
            <div class="xyz-box"><span class="xyz-axis-z">Z</span><span id="posZ">0.23 m</span></div>
          </div>
          <div class="prop-label">Rotation</div>
          <div class="xyz-grid">
            <div class="xyz-box"><span class="xyz-axis-x">X</span><span id="rotX">0.0</span></div>
            <div class="xyz-box"><span class="xyz-axis-y">Y</span><span id="rotY">0.0</span></div>
            <div class="xyz-box"><span class="xyz-axis-z">Z</span><span id="rotZ">90.0 °</span></div>
          </div>
          <div class="prop-label">Scale</div>
          <div class="xyz-grid" style="margin-bottom:0;">
            <div class="xyz-box"><span class="xyz-axis-x">X</span><span>1.0</span></div>
            <div class="xyz-box"><span class="xyz-axis-y">Y</span><span>1.0</span></div>
            <div class="xyz-box"><span class="xyz-axis-z">Z</span><span>1.0</span></div>
          </div>
        </div>

        <!-- Status Section -->
        <div class="insp-section">
          <div class="sec-title"><span>Status</span></div>
          <div class="kv-row"><span class="kv-key">Battery</span><span class="kv-val" id="batteryVal">85%</span></div>
          <div class="progress-track"><div class="progress-fill-green" style="width:85%;"></div></div>
          <div class="kv-row"><span class="kv-key">Speed</span><span class="kv-val" id="speedVal">1.2 m/s</span></div>
          <div class="kv-row"><span class="kv-key">Mode</span><span class="kv-val" id="modeVal">Autonomous</span></div>
          <div class="kv-row"><span class="kv-key">State</span><span class="kv-val" id="stateVal" style="color:#4ade80;">Navigating</span></div>
        </div>

        <!-- Navigation & Occupancy Minimap Section -->
        <div class="insp-section">
          <div class="sec-title"><span>Navigation</span></div>
          <div class="kv-row"><span class="kv-key">Destination</span><span class="kv-val">Goal_01</span></div>
          <div class="kv-row"><span class="kv-key">Distance</span><span class="kv-val" id="navDistVal">18.6 m</span></div>
          <div class="kv-row"><span class="kv-key">ETA</span><span class="kv-val" id="navEtaVal">00:00:24</span></div>
          <div class="kv-row">
            <span class="kv-key">Path</span>
            <span style="color:#3b82f6;cursor:pointer;font-size:11px;" onclick="toggleCosmosDreamMode()">Edit / Audit</span>
          </div>
          <canvas id="minimapCanvas" width="230" height="105"></canvas>
        </div>

        <!-- Simulation Time Card -->
        <div class="insp-section">
          <div class="sec-title"><span>Simulation Time</span></div>
          <div class="kv-row" style="font-size:13px;font-weight:600;padding-bottom:5px;">
            <span>🕒 <span id="simClockCard">00:01:23</span></span>
          </div>
          <div class="kv-row"><span class="kv-key">Sim Rate</span><span class="kv-val">1.00x</span></div>
          <div class="progress-track"><div class="progress-fill-green" style="width:92%;"></div></div>
          <div class="kv-row"><span class="kv-key">Delta Time</span><span class="kv-val">0.016 s</span></div>
        </div>
      </div>
    </div>

    <!-- BOTTOM LEFT: CONSOLE / ASSETS / TASKS -->
    <div class="panel console-panel">
      <div class="panel-header" style="justify-content:flex-start;gap:16px;">
        <span style="color:#fff;border-bottom:2px solid #3b82f6;padding:5px 0;">Console</span>
        <span style="color:#9ca3af;font-weight:400;cursor:pointer;">Assets</span>
        <span style="color:#9ca3af;font-weight:400;cursor:pointer;">Tasks</span>
      </div>
      <div class="console-logs" id="consoleBox">
        <div><span class="log-time">10:21:15</span> <span class="log-info">[INFO]</span> Simulation started</div>
        <div><span class="log-time">10:21:16</span> <span class="log-info">[INFO]</span> Robot_01 initialized</div>
        <div><span class="log-time">10:21:17</span> <span class="log-info">[INFO]</span> Lidar_01 online</div>
        <div><span class="log-time">10:21:18</span> <span class="log-info">[INFO]</span> Map loaded</div>
        <div><span class="log-time">10:21:19</span> <span class="log-info">[INFO]</span> Navigation goal set</div>
        <div><span class="log-time">10:21:19</span> <span class="log-info">[INFO]</span> Robot_01 navigating</div>
        <div><span class="log-time">10:21:20</span> <span class="log-info">[AUDIT]</span> Nebius Token Factory (Nemotron-3-Nano-30B-A3B) + Tavily ready</div>
      </div>
    </div>

    <!-- BOTTOM CENTER & RIGHT: SENSORS STRIP + METRICS -->
    <div class="bottom-center-grid">
      <div class="panel">
        <div class="panel-header"><span>Sensors</span></div>
        <div class="sensors-strip">
          <div class="sensor-card">
            <div class="sensor-title"><span>Lidar_01</span><span style="color:#4ade80;">360° PointCloud</span></div>
            <canvas id="lidarCanvas" class="sensor-canvas" width="220" height="125"></canvas>
          </div>
          <div class="sensor-card">
            <div class="sensor-title"><span>Camera_Front</span><span style="color:#38bdf8;">RGB 60fps</span></div>
            <canvas id="camFrontCanvas" class="sensor-canvas" width="220" height="125"></canvas>
          </div>
          <div class="sensor-card">
            <div class="sensor-title"><span>Camera_Rear</span><span style="color:#38bdf8;">RGB 60fps</span></div>
            <canvas id="camRearCanvas" class="sensor-canvas" width="220" height="125"></canvas>
          </div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-header"><span>Metrics</span></div>
        <div class="metrics-body">
          <div>
            <div class="kv-row"><span class="kv-key">CPU</span><span class="kv-val" id="cpuMetric">32%</span></div>
            <div class="progress-track"><div class="progress-fill-blue" id="cpuBar" style="width:32%;"></div></div>
          </div>
          <div>
            <div class="kv-row"><span class="kv-key">GPU</span><span class="kv-val" id="gpuMetric">48%</span></div>
            <div class="progress-track"><div class="progress-fill-blue" id="gpuBar" style="width:48%;"></div></div>
          </div>
          <div>
            <div class="kv-row"><span class="kv-key">Memory</span><span class="kv-val">7.2 / 16 GB</span></div>
            <div class="progress-track"><div class="progress-fill-blue" style="width:45%;"></div></div>
          </div>
          <div class="kv-row" style="padding-top:2px;">
            <span class="kv-key">FPS</span>
            <span class="kv-val" id="fpsMetric" style="color:#4ade80;">60.1</span>
          </div>
        </div>
      </div>
    </div>

  </div>

<script>
const AUDIT_REPORTS = {reports_json_str};

let state = {{
  playing: true,
  frame: 83,
  maxFrames: 240,
  scenario: 'warehouse_amr',
  cosmosDreamMode: false,
  cameraPreset: 'perspective',
  yawOffset: 0.0,
  pitchOffset: 0.0,
  zoom: 1.0,
  dragging: false,
  lastMouseX: 0,
  lastMouseY: 0,
  showBBoxes: true,
  layers: {{
    warehouse: true,
    environment: true,
    floor: true,
    lighting: true,
    walls: true,
    racks: true,
    robots: true,
    robot_01: true,
    forklift_01: true,
    objects: true,
    lidar: true,
    cameras: true
  }}
}};

const mainCanvas = document.getElementById('main3dCanvas');
const ctx = mainCanvas.getContext('2d');

// Orbit drag handlers on 3D viewport
mainCanvas.addEventListener('mousedown', (e) => {{
  state.dragging = true;
  state.lastMouseX = e.clientX;
  state.lastMouseY = e.clientY;
}});
window.addEventListener('mouseup', () => {{ state.dragging = false; }});
window.addEventListener('mousemove', (e) => {{
  if (!state.dragging) return;
  const dx = e.clientX - state.lastMouseX;
  const dy = e.clientY - state.lastMouseY;
  state.lastMouseX = e.clientX;
  state.lastMouseY = e.clientY;
  state.yawOffset = Math.max(-0.65, Math.min(0.65, state.yawOffset + dx * 0.004));
  state.pitchOffset = Math.max(-0.35, Math.min(0.45, state.pitchOffset + dy * 0.003));
}});
mainCanvas.addEventListener('wheel', (e) => {{
  e.preventDefault();
  state.zoom = Math.max(0.65, Math.min(1.65, state.zoom - e.deltaY * 0.0008));
}}, {{ passive: false }});

function resizeCanvases() {{
  const rect = mainCanvas.parentElement.getBoundingClientRect();
  mainCanvas.width = Math.max(400, Math.floor(rect.width));
  mainCanvas.height = Math.max(260, Math.floor(rect.height));
}}
window.addEventListener('resize', resizeCanvases);
resizeCanvases();

function appendConsoleLog(level, msg) {{
  const box = document.getElementById('consoleBox');
  const now = new Date();
  const ts = now.toTimeString().split(' ')[0];
  const cls = level === 'ERR' ? 'log-err' : (level === 'WARN' ? 'log-warn' : 'log-info');
  const div = document.createElement('div');
  div.innerHTML = `<span class="log-time">${{ts}}</span> <span class="${{cls}}">[${{level}}]</span> ${{msg}}`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}}

function togglePlay() {{
  state.playing = !state.playing;
  const icon = state.playing ? '⏸' : '▶';
  document.getElementById('tlPlayBtn').textContent = icon;
  document.getElementById('vpPlayPauseBtn').textContent = icon;
  appendConsoleLog('INFO', state.playing ? 'Simulation resumed (1.00x)' : 'Simulation paused');
}}

function restartSim() {{
  state.frame = 0;
  document.getElementById('timelineSlider').value = 0;
  appendConsoleLog('INFO', 'Timeline reset to 00:00:00');
}}

function stepFrame() {{
  state.playing = false;
  state.frame = (state.frame + 1) % (state.maxFrames + 1);
  document.getElementById('timelineSlider').value = state.frame;
  document.getElementById('tlPlayBtn').textContent = '▶';
  document.getElementById('vpPlayPauseBtn').textContent = '▶';
}}

function scrubTimeline(val) {{
  state.frame = parseInt(val, 10);
}}

function resetCamera() {{
  state.yawOffset = 0;
  state.pitchOffset = 0;
  state.zoom = 1.0;
  state.cameraPreset = 'perspective';
  document.getElementById('cameraViewSelect').value = 'perspective';
}}

function setCameraPreset(preset) {{
  state.cameraPreset = preset;
  if (preset === 'top') {{
    state.pitchOffset = 0.42;
    state.yawOffset = 0.0;
  }} else if (preset === 'chase') {{
    state.pitchOffset = -0.08;
    state.yawOffset = 0.0;
    state.zoom = 1.18;
  }} else {{
    state.pitchOffset = 0.0;
    state.yawOffset = 0.0;
    state.zoom = 1.0;
  }}
  appendConsoleLog('INFO', `Viewport camera switched to ${{preset}}`);
}}

function toggleBBoxes() {{
  state.showBBoxes = !state.showBBoxes;
  appendConsoleLog('INFO', `3D spatial bounding boxes: ${{state.showBBoxes ? 'ON' : 'OFF'}}`);
}}

function toggleLayer(layerKey, btnEl) {{
  state.layers[layerKey] = !state.layers[layerKey];
  if (btnEl) btnEl.classList.toggle('hidden-node', !state.layers[layerKey]);
  appendConsoleLog('INFO', `Scene layer '${{layerKey}}' visibility=${{state.layers[layerKey]}}`);
}}

function filterSceneTree(query) {{
  const q = query.trim().toLowerCase();
  document.querySelectorAll('#sceneTree .tree-row').forEach(row => {{
    const name = row.getAttribute('data-name') || '';
    row.style.display = (!q || name.includes(q)) ? 'flex' : 'none';
  }});
}}

function selectEntity(name, rowEl) {{
  document.querySelectorAll('#sceneTree .tree-row').forEach(r => r.classList.remove('selected'));
  if (rowEl) rowEl.classList.add('selected');
  document.getElementById('inspectorEntityTitle').textContent = name;
  appendConsoleLog('INFO', `Selected scene entity '${{name}}'`);
}}

function switchTopTab(el, tabName) {{
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  if (tabName === 'Diagnostics') {{
    if (!state.cosmosDreamMode) toggleCosmosDreamMode();
  }}
  appendConsoleLog('INFO', `Switched workspace view to ${{tabName}}`);
}}

function switchInspTab(el) {{
  document.querySelectorAll('.insp-tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
}}

function changeScenario(scen) {{
  state.scenario = scen;
  const rep = AUDIT_REPORTS[scen] || AUDIT_REPORTS['warehouse_amr'];
  updateAuditHeader(rep);
  appendConsoleLog('INFO', `Loaded 3D scenario '${{scen}}'`);
}}

function toggleCosmosDreamMode() {{
  state.cosmosDreamMode = !state.cosmosDreamMode;
  const btn = document.getElementById('cosmosModeBtn');
  const dot = document.getElementById('cosmosModeDot');
  const lbl = document.getElementById('cosmosModeLabel');
  const rep = AUDIT_REPORTS[state.scenario] || {{ physics_hallucination_rate: 0.218, physical_integrity_rating: 0.614 }};

  if (state.cosmosDreamMode) {{
    btn.classList.add('hallucination-active');
    dot.style.color = '#ef4444';
    lbl.textContent = 'Mode: Cosmos 3D Dream (Audit Overlay)';
    document.getElementById('phrVal').textContent = rep.physics_hallucination_rate.toFixed(3);
    document.getElementById('phrVal').style.color = '#f87171';
    document.getElementById('pirVal').textContent = rep.physical_integrity_rating.toFixed(3);
    document.getElementById('stateVal').textContent = 'Physics Violation';
    document.getElementById('stateVal').style.color = '#f87171';
    appendConsoleLog('ERR', `[NEMOTRON AUDIT] Cosmos 3D Dream PHR=${{rep.physics_hallucination_rate.toFixed(3)}} (FAIL): Contact & Friction violations detected`);
  }} else {{
    btn.classList.remove('hallucination-active');
    dot.style.color = '#4ade80';
    lbl.textContent = 'Mode: Isaac Lab 3D Ground Truth';
    document.getElementById('phrVal').textContent = '0.000';
    document.getElementById('phrVal').style.color = '#4ade80';
    document.getElementById('pirVal').textContent = '0.992';
    document.getElementById('stateVal').textContent = 'Navigating';
    document.getElementById('stateVal').style.color = '#4ade80';
    appendConsoleLog('INFO', `Switched to NVIDIA Isaac Lab 3D Ground Truth Oracle (PHR=0.000)`);
  }}
}}

function updateAuditHeader(rep) {{
  if (!rep) return;
  if (state.cosmosDreamMode) {{
    document.getElementById('phrVal').textContent = (rep.physics_hallucination_rate || 0.218).toFixed(3);
    document.getElementById('pirVal').textContent = (rep.physical_integrity_rating || 0.614).toFixed(3);
  }}
}}

// 3D Perspective Projection Helper for Warehouse Aisle
function project3D(wx, wy, wz, w, h) {{
  // wx: lateral across aisle (-6..+6m), wy: depth down corridor (1..36m), wz: height (0..7m)
  const u = state.frame / state.maxFrames;
  const camFollowY = state.cameraPreset === 'chase' ? (u * 10.0) : 0.0;

  // Apply interactive orbit offsets
  const cosY = Math.cos(state.yawOffset);
  const sinY = Math.sin(state.yawOffset);
  const rx = wx * cosY - (wy - 8.0) * sinY;
  const ry = (wx * sinY + (wy - 8.0) * cosY) + 8.0 - camFollowY;

  const camX = 0.0;
  const camY = -1.8;
  const camZ = (state.cameraPreset === 'top' ? 9.5 : 2.35) + state.pitchOffset * 5.0;

  const dz = Math.max(0.8, ry - camY);
  const fov = 520.0 * state.zoom * (w / 960.0);
  const horizonY = h * (state.cameraPreset === 'top' ? 0.18 : 0.34) + state.pitchOffset * 160.0;

  const sx = w * 0.50 + ((rx - camX) / dz) * fov;
  const sy = horizonY - ((wz - camZ) / dz) * fov;
  return {{ x: sx, y: sy, depth: dz, scale: fov / dz }};
}}

function drawQuad3D(p0, p1, p2, p3, fillStyle, strokeStyle, lineWidth = 1) {{
  ctx.beginPath();
  ctx.moveTo(p0.x, p0.y);
  ctx.lineTo(p1.x, p1.y);
  ctx.lineTo(p2.x, p2.y);
  ctx.lineTo(p3.x, p3.y);
  ctx.closePath();
  if (fillStyle) {{
    ctx.fillStyle = fillStyle;
    ctx.fill();
  }}
  if (strokeStyle) {{
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
  }}
}}

function drawBox3D(x0, x1, y0, y1, z0, z1, w, h, colors) {{
  const b000 = project3D(x0, y0, z0, w, h);
  const b100 = project3D(x1, y0, z0, w, h);
  const b110 = project3D(x1, y1, z0, w, h);
  const b010 = project3D(x0, y1, z0, w, h);

  const t001 = project3D(x0, y0, z1, w, h);
  const t101 = project3D(x1, y0, z1, w, h);
  const t111 = project3D(x1, y1, z1, w, h);
  const t011 = project3D(x0, y1, z1, w, h);

  // Inner aisle-facing side
  if (x1 < 0) {{
    drawQuad3D(b100, b110, t111, t101, colors.side, colors.stroke, 1);
  }} else if (x0 > 0) {{
    drawQuad3D(b000, b010, t011, t001, colors.side, colors.stroke, 1);
  }}
  // Top face
  drawQuad3D(t001, t101, t111, t011, colors.top, colors.stroke, 1);
  // Front face (facing camera at y0)
  drawQuad3D(b000, b100, t101, t001, colors.front, colors.stroke, 1);
}}

function drawWireBox3D(x0, x1, y0, y1, z0, z1, w, h, strokeColor, fillAlphaColor) {{
  const c = [
    project3D(x0, y0, z0, w, h), project3D(x1, y0, z0, w, h),
    project3D(x1, y1, z0, w, h), project3D(x0, y1, z0, w, h),
    project3D(x0, y0, z1, w, h), project3D(x1, y0, z1, w, h),
    project3D(x1, y1, z1, w, h), project3D(x0, y1, z1, w, h)
  ];
  if (fillAlphaColor) {{
    drawQuad3D(c[0], c[1], c[5], c[4], fillAlphaColor, null);
    drawQuad3D(c[4], c[5], c[6], c[7], fillAlphaColor, null);
  }}
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 1.4;
  // Bottom & Top loops + vertical pillars
  const edges = [
    [0,1],[1,2],[2,3],[3,0],
    [4,5],[5,6],[6,7],[7,4],
    [0,4],[1,5],[2,6],[3,7]
  ];
  ctx.beginPath();
  edges.forEach(([a, b]) => {{
    ctx.moveTo(c[a].x, c[a].y);
    ctx.lineTo(c[b].x, c[b].y);
  }});
  ctx.stroke();
}}

function getSplinePoint(tNorm) {{
  // Smooth warehouse aisle S-curve spline from y=3.2m to y=26.0m
  const wy = 3.2 + tNorm * 22.0;
  const wx = -0.65 * Math.cos(tNorm * Math.PI * 1.15) + 0.32 * Math.sin(tNorm * Math.PI * 2.4);
  return {{ x: wx, y: wy, z: 0.0 }};
}}

function renderMain3DViewport() {{
  const w = mainCanvas.width;
  const h = mainCanvas.height;
  ctx.clearRect(0, 0, w, h);

  // 1. Warehouse Ceiling & Back Wall Atmosphere
  const ceilGrad = ctx.createLinearGradient(0, 0, 0, h * 0.45);
  ceilGrad.addColorStop(0, '#1c2026');
  ceilGrad.addColorStop(1, '#2b303a');
  ctx.fillStyle = ceilGrad;
  ctx.fillRect(0, 0, w, h * 0.45);

  // Industrial back warehouse wall at y = 32m
  if (state.layers.walls) {{
    const w0 = project3D(-9, 32, 0, w, h);
    const w1 = project3D(9, 32, 0, w, h);
    const w2 = project3D(9, 32, 7.5, w, h);
    const w3 = project3D(-9, 32, 7.5, w, h);
    drawQuad3D(w0, w1, w2, w3, '#393f4a', '#4b5563', 1);
  }}

  // High-bay warehouse ceiling trusses & LED strip lights
  if (state.layers.lighting) {{
    for (let distY = 4; distY <= 28; distY += 5) {{
      const l0 = project3D(-1.2, distY, 6.6, w, h);
      const l1 = project3D(1.2, distY, 6.6, w, h);
      ctx.strokeStyle = 'rgba(248, 250, 252, 0.75)';
      ctx.lineWidth = Math.max(1.5, 18 / l0.depth);
      ctx.beginPath();
      ctx.moveTo(l0.x, l0.y);
      ctx.lineTo(l1.x, l1.y);
      ctx.stroke();
    }}
  }}

  // 2. Polished Concrete Warehouse Floor
  if (state.layers.floor) {{
    const f0 = project3D(-9.5, 0.5, 0, w, h);
    const f1 = project3D(9.5, 0.5, 0, w, h);
    const f2 = project3D(9.5, 32, 0, w, h);
    const f3 = project3D(-9.5, 32, 0, w, h);
    const floorGrad = ctx.createLinearGradient(0, h * 0.32, 0, h);
    floorGrad.addColorStop(0, '#353941');
    floorGrad.addColorStop(0.45, '#2b2f37');
    floorGrad.addColorStop(1, '#1e2229');
    drawQuad3D(f0, f1, f2, f3, floorGrad, null);

    // Perspective concrete expansion joints & floor grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let gx = -6; gx <= 6; gx += 1.5) {{
      const pA = project3D(gx, 1.0, 0, w, h);
      const pB = project3D(gx, 32.0, 0, w, h);
      ctx.beginPath(); ctx.moveTo(pA.x, pA.y); ctx.lineTo(pB.x, pB.y); ctx.stroke();
    }}
    for (let gy = 2; gy <= 30; gy += 2.0) {{
      const pA = project3D(-6.0, gy, 0, w, h);
      const pB = project3D(6.0, gy, 0, w, h);
      ctx.beginPath(); ctx.moveTo(pA.x, pA.y); ctx.lineTo(pB.x, pB.y); ctx.stroke();
    }}

    // White aisle boundary lines
    ctx.strokeStyle = 'rgba(226, 232, 240, 0.25)';
    ctx.lineWidth = 1.5;
    [-2.55, 2.55].forEach(ax => {{
      const a0 = project3D(ax, 1.5, 0.01, w, h);
      const a1 = project3D(ax, 30.0, 0.01, w, h);
      ctx.beginPath(); ctx.moveTo(a0.x, a0.y); ctx.lineTo(a1.x, a1.y); ctx.stroke();
    }});
  }}

  // 3. Industrial Warehouse Racks (Left & Right Aisles, painter's algorithm back-to-front)
  if (state.layers.racks) {{
    const rackSegments = [
      {{ yStart: 22.0, yEnd: 27.5, leftLabel: null, rightLabel: null }},
      {{ yStart: 15.0, yEnd: 20.5, leftLabel: 'B4', rightLabel: 'B5' }},
      {{ yStart: 8.5,  yEnd: 14.0, leftLabel: 'A5', rightLabel: null }},
      {{ yStart: 2.2,  yEnd: 7.6,  leftLabel: 'A4', rightLabel: null }}
    ];

    rackSegments.forEach(seg => {{
      // Left Rack (-5.4m to -2.85m) and Right Rack (+2.85m to +5.4m)
      [{{ x0: -5.2, x1: -2.85, side: 'L' }}, {{ x0: 2.85, x1: 5.2, side: 'R' }}].forEach(rk => {{
        // Cardboard boxes on 3 shelf tiers (z = 0.2..1.6, 1.9..3.3, 3.6..5.0)
        if (state.layers.objects) {{
          [0.18, 1.95, 3.70].forEach((shelfZ, tierIdx) => {{
            const boxStep = (seg.yEnd - seg.yStart) / 2.0;
            for (let bi = 0; bi < 2; bi++) {{
              const by0 = seg.yStart + bi * boxStep + 0.15;
              const by1 = by0 + boxStep - 0.30;
              drawBox3D(
                rk.x0 + 0.15, rk.x1 - 0.15,
                by0, by1,
                shelfZ, shelfZ + 1.32,
                w, h,
                {{
                  front: tierIdx % 2 === 0 ? '#8c6747' : '#9c7654',
                  side: '#735338',
                  top: '#a88463',
                  stroke: '#4a3523'
                }}
              );
            }}
          }});
        }}

        // Orange horizontal rack beams
        [1.75, 3.50, 5.15].forEach(beamZ => {{
          const faceX = rk.side === 'L' ? rk.x1 : rk.x0;
          const bA = project3D(faceX, seg.yStart, beamZ, w, h);
          const bB = project3D(faceX, seg.yEnd, beamZ, w, h);
          ctx.strokeStyle = '#d97706';
          ctx.lineWidth = Math.max(2, 22 / bA.depth);
          ctx.beginPath(); ctx.moveTo(bA.x, bA.y); ctx.lineTo(bB.x, bB.y); ctx.stroke();
        }});

        // Vertical steel uprights + yellow base guards
        [seg.yStart, seg.yEnd].forEach(vy => {{
          const faceX = rk.side === 'L' ? rk.x1 : rk.x0;
          const uBot = project3D(faceX, vy, 0, w, h);
          const uMid = project3D(faceX, vy, 0.85, w, h);
          const uTop = project3D(faceX, vy, 5.5, w, h);
          ctx.strokeStyle = '#475569';
          ctx.lineWidth = Math.max(2, 20 / uBot.depth);
          ctx.beginPath(); ctx.moveTo(uBot.x, uBot.y); ctx.lineTo(uTop.x, uTop.y); ctx.stroke();

          // Yellow safety post guard at bottom
          ctx.strokeStyle = '#eab308';
          ctx.lineWidth = Math.max(3, 28 / uBot.depth);
          ctx.beginPath(); ctx.moveTo(uBot.x, uBot.y); ctx.lineTo(uMid.x, uMid.y); ctx.stroke();
        }});
      }});

      // Aisle Markers: A4, A5, B4, B5 white signage plates
      if (seg.leftLabel) {{
        const signPos = project3D(-2.82, seg.yStart + 0.6, 2.55, w, h);
        const sw = Math.max(22, 260 / signPos.depth);
        const sh = sw * 1.15;
        ctx.fillStyle = '#f1f5f9';
        ctx.fillRect(signPos.x - sw * 0.5, signPos.y - sh * 0.5, sw, sh);
        ctx.strokeStyle = '#1e293b';
        ctx.strokeRect(signPos.x - sw * 0.5, signPos.y - sh * 0.5, sw, sh);
        ctx.fillStyle = '#0f172a';
        ctx.font = `bold ${{Math.max(10, Math.floor(sw * 0.52))}}px Inter, sans-serif`;
        ctx.textAlign = 'center';
        ctx.fillText(seg.leftLabel, signPos.x, signPos.y + sh * 0.18);
      }}
      if (seg.rightLabel) {{
        const signPos = project3D(-0.85, seg.yStart + 1.2, 2.65, w, h);
        const sw = Math.max(20, 230 / signPos.depth);
        const sh = sw * 1.12;
        ctx.fillStyle = '#f1f5f9';
        ctx.fillRect(signPos.x - sw * 0.5, signPos.y - sh * 0.5, sw, sh);
        ctx.fillStyle = '#0f172a';
        ctx.font = `bold ${{Math.max(9, Math.floor(sw * 0.5))}}px Inter, sans-serif`;
        ctx.textAlign = 'center';
        ctx.fillText(seg.rightLabel, signPos.x, signPos.y + sh * 0.18);
      }}
    }});
  }}

  // 4. Right-Side Obstacle Pallets with Purple 3D Detection Wireframes
  if (state.layers.objects && state.showBBoxes) {{
    drawBox3D(1.55, 2.45, 11.2, 12.6, 0.0, 0.22, w, h, {{
      front: '#991b1b', side: '#7f1d1d', top: '#b91c1c', stroke: '#ef4444'
    }});
    drawWireBox3D(1.45, 2.55, 11.0, 12.8, 0.0, 1.35, w, h, 'rgba(168, 85, 247, 0.85)', 'rgba(168, 85, 247, 0.14)');
  }}

  // 5. Forklift 01 (Yellow/Black Industrial Forklift at y = 15.6m)
  if (state.layers.forklift_01) {{
    const fkY = 15.6;
    const fkX = 0.95;
    // Forklift lower counterweight body
    drawBox3D(fkX - 0.55, fkX + 0.55, fkY, fkY + 1.9, 0.08, 0.95, w, h, {{
      front: '#ca8a04', side: '#eab308', top: '#facc15', stroke: '#713f12'
    }});
    // Roll cage cab
    drawWireBox3D(fkX - 0.48, fkX + 0.48, fkY + 0.2, fkY + 1.5, 0.95, 2.15, w, h, '#1e293b', 'rgba(15, 23, 42, 0.35)');
    // Vertical mast
    const m0 = project3D(fkX - 0.3, fkY, 0.1, w, h);
    const m1 = project3D(fkX - 0.3, fkY, 2.6, w, h);
    const m2 = project3D(fkX + 0.3, fkY, 0.1, w, h);
    const m3 = project3D(fkX + 0.3, fkY, 2.6, w, h);
    ctx.strokeStyle = '#0f172a'; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(m0.x, m0.y); ctx.lineTo(m1.x, m1.y); ctx.moveTo(m2.x, m2.y); ctx.lineTo(m3.x, m3.y); ctx.stroke();

    if (state.showBBoxes) {{
      drawWireBox3D(-0.15, 0.75, 12.5, 14.3, 0.0, 0.95, w, h, 'rgba(239, 68, 68, 0.75)', 'rgba(239, 68, 68, 0.10)');
    }}

    // Floating Tag: Forklift 01
    const fkTag = project3D(fkX - 0.8, fkY + 0.4, 1.55, w, h);
    drawHudTag(fkTag.x, fkTag.y, 'Forklift 01', '#4ade80');
  }}

  // 6. Green -> Cyan -> Purple Navigation Spline & Waypoint Nodes on Floor
  const totalSplineSteps = 60;
  for (let i = 0; i < totalSplineSteps; i++) {{
    const t0 = i / totalSplineSteps;
    const t1 = (i + 1) / totalSplineSteps;
    const p0 = getSplinePoint(t0);
    const p1 = getSplinePoint(t1);
    const s0 = project3D(p0.x, p0.y, 0.03, w, h);
    const s1 = project3D(p1.x, p1.y, 0.03, w, h);

    ctx.strokeStyle = t0 > 0.72 ? '#a855f7' : '#4ade80';
    ctx.lineWidth = Math.max(2.0, 24 / s0.depth);
    ctx.beginPath();
    ctx.moveTo(s0.x, s0.y);
    ctx.lineTo(s1.x, s1.y);
    ctx.stroke();
  }}

  // Waypoint discs along path
  [0.06, 0.22, 0.38, 0.55, 0.72, 0.88, 1.0].forEach((wt, idx) => {{
    const wp = getSplinePoint(wt);
    const sp = project3D(wp.x, wp.y, 0.04, w, h);
    const rad = Math.max(3, 42 / sp.depth);
    ctx.fillStyle = wt === 1.0 ? '#a855f7' : (idx < 2 ? '#4ade80' : '#3b82f6');
    ctx.beginPath();
    ctx.arc(sp.x, sp.y, rad, 0, Math.PI * 2);
    ctx.fill();
  }});

  // Goal Flag at end of spline
  const goalPt = getSplinePoint(1.0);
  const goalScreen = project3D(goalPt.x, goalPt.y, 1.15, w, h);
  drawHudTag(goalScreen.x, goalScreen.y, 'Goal', '#a855f7');

  // 7. Robot 01 Position & 360° LiDAR Ray Fan
  const uRobot = 0.14 + (state.frame / state.maxFrames) * 0.55;
  const rPos = getSplinePoint(uRobot);

  // If Cosmos 3D Dream Mode is active, inject physical hallucination drift (lateral rack collision / levitation)
  let hallucinationType = null;
  if (state.cosmosDreamMode) {{
    if (state.frame >= 60 && state.frame <= 115) {{
      rPos.x -= 1.45 * Math.sin(((state.frame - 60) / 55.0) * Math.PI);
      hallucinationType = 'Contact Constraint Violation: Rack_A SDF = -0.22m';
    }} else if (state.frame >= 140 && state.frame <= 185) {{
      rPos.z = 0.55 * Math.sin(((state.frame - 140) / 45.0) * Math.PI);
      hallucinationType = 'Gravity & Floor Contact Violation: Unforced Levitation +0.55m';
    }} else {{
      rPos.x += 0.35 * Math.sin(state.frame * 0.2);
      hallucinationType = 'Non-Holonomic Tire Friction Slip (mu=0.72 Exceeded)';
    }}
  }}

  // Draw 360° LiDAR Rays & Point-Cloud Floor Hits around Robot 01
  if (state.layers.lidar && state.layers.robot_01) {{
    const lidarOrigin = project3D(rPos.x, rPos.y, rPos.z + 0.72, w, h);
    const numRays = 56;
    for (let i = 0; i < numRays; i++) {{
      const ang = (i / numRays) * Math.PI * 2 + state.frame * 0.03;
      const maxRange = 3.6 + 0.8 * Math.sin(i * 1.7);
      let hitX = rPos.x + Math.cos(ang) * maxRange;
      let hitY = rPos.y + Math.sin(ang) * maxRange * 1.35;
      hitX = Math.max(-2.82, Math.min(2.82, hitX));
      const hitZ = (Math.abs(hitX) >= 2.75) ? (0.35 + (i % 4) * 0.28) : 0.03;

      const hitScreen = project3D(hitX, hitY, hitZ, w, h);
      ctx.strokeStyle = state.cosmosDreamMode
        ? 'rgba(248, 113, 113, 0.25)'
        : 'rgba(56, 189, 248, 0.22)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(lidarOrigin.x, lidarOrigin.y);
      ctx.lineTo(hitScreen.x, hitScreen.y);
      ctx.stroke();

      // Glowing laser hit dot
      ctx.fillStyle = Math.abs(hitX) >= 2.75 ? '#4ade80' : '#38bdf8';
      ctx.fillRect(hitScreen.x - 1.5, hitScreen.y - 1.5, 3, 3);
    }}
  }}

  // 8. Draw Robot 01 (Autonomous Mobile Robot Chassis + White Upper Housing + Spinning LiDAR Turret)
  if (state.layers.robot_01) {{
    const rx = rPos.x;
    const ry = rPos.y;
    const rz = rPos.z;

    // Lower dark bumper & drive chassis
    drawBox3D(rx - 0.56, rx + 0.56, ry - 0.72, ry + 0.72, rz + 0.04, rz + 0.34, w, h, {{
      front: '#18181b', side: '#27272a', top: '#3f3f46', stroke: '#09090b'
    }});
    // Upper white fiberglass body shell
    drawBox3D(rx - 0.52, rx + 0.52, ry - 0.66, ry + 0.66, rz + 0.34, rz + 0.78, w, h, {{
      front: '#e2e8f0', side: '#cbd5e1', top: '#f8fafc', stroke: '#64748b'
    }});
    // Dark sensor band & LiDAR puck on roof
    drawBox3D(rx - 0.14, rx + 0.14, ry - 0.14, ry + 0.14, rz + 0.78, rz + 0.94, w, h, {{
      front: '#0f172a', side: '#1e293b', top: '#38bdf8', stroke: '#475569'
    }});

    // Rear red LED tail light dot
    const ledPt = project3D(rx, ry - 0.73, rz + 0.22, w, h);
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.arc(ledPt.x, ledPt.y, 3.5, 0, Math.PI * 2);
    ctx.fill();

    // Floating Tag: Robot 01
    const robTag = project3D(rx - 0.92, ry - 0.2, rz + 0.65, w, h);
    drawHudTag(robTag.x, robTag.y, 'Robot 01', state.cosmosDreamMode ? '#ef4444' : '#4ade80');

    // If Cosmos Hallucination active, draw red 3D collision bounding box around Robot 01
    if (state.cosmosDreamMode) {{
      drawWireBox3D(rx - 0.68, rx + 0.68, ry - 0.85, ry + 0.85, rz, rz + 1.05, w, h, '#ef4444', 'rgba(239, 68, 68, 0.18)');
      if (hallucinationType) {{
        ctx.fillStyle = 'rgba(127, 29, 29, 0.92)';
        ctx.strokeStyle = '#f87171';
        ctx.lineWidth = 1.5;
        ctx.fillRect(14, 46, 430, 28);
        ctx.strokeRect(14, 46, 430, 28);
        ctx.fillStyle = '#fef2f2';
        ctx.font = '600 11.5px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`⚠️ NEMOTRON 3D AUDIT: ${{hallucinationType}}`, 24, 64);
      }}
    }}
  }}

  // Update Inspector Telemetry
  const worldX = (12.42 + rPos.x * 0.45).toFixed(2);
  const worldY = (4.85 + (rPos.y - 6.0) * 0.35).toFixed(2);
  const worldZ = (0.23 + rPos.z).toFixed(2);
  document.getElementById('posX').textContent = worldX;
  document.getElementById('posY').textContent = worldY;
  document.getElementById('posZ').textContent = `${{worldZ}} m`;

  const remDist = Math.max(0.0, 24.5 - uRobot * 22.0).toFixed(1);
  document.getElementById('navDistVal').textContent = `${{remDist}} m`;
  const etaSec = Math.max(0, Math.round(remDist / 1.2));
  document.getElementById('navEtaVal').textContent = `00:00:${{String(etaSec).padStart(2, '0')}}`;

  const totalSec = Math.floor(state.frame);
  const mm = String(Math.floor(totalSec / 60)).padStart(2, '0');
  const ss = String(totalSec % 60).padStart(2, '0');
  const tc = `00:${{mm}}:${{ss}}`;
  document.getElementById('timecodeDisplay').textContent = `${{tc}} / 00:05:00`;
  document.getElementById('simClockCard').textContent = tc;
  document.getElementById('frameCounterLabel').textContent = `Frame ${{String(state.frame).padStart(3, '0')}} / ${{state.maxFrames}}`;
}}

function drawHudTag(x, y, text, dotColor) {{
  ctx.font = '500 11px Inter, sans-serif';
  const tw = ctx.measureText(text).width + 24;
  const th = 22;
  ctx.fillStyle = 'rgba(20, 24, 33, 0.88)';
  ctx.strokeStyle = 'rgba(71, 85, 105, 0.8)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x - tw * 0.5, y - th * 0.5, tw, th, 4);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = dotColor;
  ctx.beginPath();
  ctx.arc(x - tw * 0.5 + 9, y, 3.2, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = '#f3f4f6';
  ctx.textAlign = 'left';
  ctx.fillText(text, x - tw * 0.5 + 16, y + 4);
}}

// Render Bottom Sensor 1: Lidar_01 Point Cloud
const lidarCanvas = document.getElementById('lidarCanvas');
const lctx = lidarCanvas.getContext('2d');
function renderLidarSensor() {{
  const w = lidarCanvas.width;
  const h = lidarCanvas.height;
  lctx.fillStyle = '#080b11';
  lctx.fillRect(0, 0, w, h);

  const cx = w * 0.5;
  const cy = h * 0.82;
  for (let ring = 1; ring <= 5; ring++) {{
    const r = ring * 18;
    for (let a = -Math.PI * 0.92; a <= -Math.PI * 0.08; a += 0.07) {{
      const jitter = Math.sin(a * 9 + state.frame * 0.08 + ring) * 3.5;
      const px = cx + Math.cos(a) * (r + jitter);
      const py = cy + Math.sin(a) * (r * 0.72 + jitter);
      lctx.fillStyle = ring < 3 ? '#38bdf8' : '#4ade80';
      lctx.fillRect(px, py, 2, 2);
    }}
  }}
  // Left & Right aisle wall point clusters
  for (let y = 12; y < h - 10; y += 4) {{
    const spread = (h - y) * 0.48;
    lctx.fillStyle = '#22c55e';
    lctx.fillRect(cx - spread + Math.sin(y + state.frame * 0.1) * 3, y, 2.5, 2.5);
    lctx.fillRect(cx + spread + Math.cos(y + state.frame * 0.1) * 3, y, 2.5, 2.5);
  }}
}}

// Render Bottom Sensor 2 & 3: Camera_Front & Camera_Rear
const camFront = document.getElementById('camFrontCanvas');
const cfCtx = camFront.getContext('2d');
const camRear = document.getElementById('camRearCanvas');
const crCtx = camRear.getContext('2d');

function renderMiniCamera(cctx, isRear) {{
  const w = 220, h = 125;
  cctx.fillStyle = '#1f242d';
  cctx.fillRect(0, 0, w, h * 0.45);
  cctx.fillStyle = '#2d323b';
  cctx.fillRect(0, h * 0.45, w, h * 0.55);

  // Perspective corridor racks on left and right
  cctx.fillStyle = '#7c5836';
  cctx.beginPath();
  cctx.moveTo(0, 10); cctx.lineTo(w * 0.34, h * 0.44); cctx.lineTo(w * 0.34, h * 0.62); cctx.lineTo(0, h * 0.92);
  cctx.fill();

  cctx.beginPath();
  cctx.moveTo(w, 10); cctx.lineTo(w * 0.66, h * 0.44); cctx.lineTo(w * 0.66, h * 0.62); cctx.lineTo(w, h * 0.92);
  cctx.fill();

  // Orange rack beams
  cctx.strokeStyle = '#d97706';
  cctx.lineWidth = 1.5;
  cctx.beginPath();
  cctx.moveTo(0, 38); cctx.lineTo(w * 0.34, h * 0.48);
  cctx.moveTo(w, 38); cctx.lineTo(w * 0.66, h * 0.48);
  cctx.stroke();

  // Yellow safety lines
  cctx.strokeStyle = '#eab308';
  cctx.beginPath();
  cctx.moveTo(18, h); cctx.lineTo(w * 0.38, h * 0.58);
  cctx.moveTo(w - 18, h); cctx.lineTo(w * 0.62, h * 0.58);
  cctx.stroke();

  if (isRear) {{
    // Yellow Forklift visible in rear camera
    const fx = w * 0.62 + Math.sin(state.frame * 0.03) * 3;
    cctx.fillStyle = '#eab308';
    cctx.fillRect(fx, h * 0.46, 26, 22);
    cctx.strokeStyle = '#1e293b';
    cctx.strokeRect(fx + 3, h * 0.34, 20, 14);
  }}
}}

// Render Right-Side 2D Top-Down Occupancy Minimap
const minimap = document.getElementById('minimapCanvas');
const mctx = minimap.getContext('2d');
function renderMinimap() {{
  const w = minimap.width;
  const h = minimap.height;
  mctx.fillStyle = '#12161f';
  mctx.fillRect(0, 0, w, h);

  // Draw top-down warehouse rack blocks
  mctx.fillStyle = '#2d3545';
  for (let row = 0; row < 3; row++) {{
    for (let col = 0; col < 5; col++) {{
      const rx = 18 + col * 40;
      const ry = row === 0 ? 10 : (row === 1 ? 44 : 78);
      if (row !== 1 || col === 0 || col === 4) {{
        mctx.fillRect(rx, ry, 28, 16);
      }}
    }}
  }}

  // Draw spline trajectory on minimap
  mctx.strokeStyle = '#4ade80';
  mctx.lineWidth = 2;
  mctx.beginPath();
  mctx.moveTo(35, 65);
  mctx.bezierCurveTo(85, 65, 115, 75, 145, 55);
  mctx.stroke();

  mctx.strokeStyle = '#a855f7';
  mctx.beginPath();
  mctx.moveTo(145, 55);
  mctx.bezierCurveTo(165, 40, 180, 35, 198, 35);
  mctx.stroke();

  // Goal dot
  mctx.fillStyle = '#a855f7';
  mctx.beginPath(); mctx.arc(198, 35, 4, 0, Math.PI * 2); mctx.fill();

  // Live Robot 01 position dot on minimap
  const u = state.frame / state.maxFrames;
  const rx = 35 + u * 163;
  const ry = 65 - Math.sin(u * Math.PI) * 14 - u * 25;
  mctx.fillStyle = state.cosmosDreamMode ? '#ef4444' : '#38bdf8';
  mctx.beginPath(); mctx.arc(rx, ry, 4.5, 0, Math.PI * 2); mctx.fill();
}}

// Animation Loop
let lastTick = performance.now();
function animate(now) {{
  if (state.playing && now - lastTick > 45) {{
    state.frame = (state.frame + 1) % (state.maxFrames + 1);
    document.getElementById('timelineSlider').value = state.frame;
    lastTick = now;
  }}
  renderMain3DViewport();
  renderLidarSensor();
  renderMiniCamera(cfCtx, false);
  renderMiniCamera(crCtx, true);
  renderMinimap();
  requestAnimationFrame(animate);
}}
requestAnimationFrame(animate);
</script>
</body>
</html>
"""


def main() -> None:
    st.set_page_config(
        page_title="Simulation Studio | Physical Fidelity Auditor",
        page_icon="🟢",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Full-bleed dark styling so the Simulation Environment Dashboard fills the viewport seamlessly
    st.markdown(
        """
        <style>
        #MainMenu, header, footer {visibility: hidden; height: 0;}
        .stApp {background-color: #0b0d11; color: #e5e7eb;}
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
            padding-left: 0rem !important;
            padding-right: 0rem !important;
            max-width: 100% !important;
        }
        iframe {display: block; border: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    all_reports = _load_all_reports()
    reports_json = json.dumps(all_reports)

    # Render the full interactive 3D Simulation Environment Dashboard
    studio_html = _build_simulation_studio_html(reports_json)
    components.html(studio_html, height=860, scrolling=False)

    # Collapsible Live Nebius Token Factory (Nemotron) + Tavily Audit Backend Drawer
    with st.expander("⚖️ Nebius Token Factory (Nemotron 3 Nano) + Tavily Physics Audit Console & Frame Inspector"):
        c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
        with c1:
            selected_scen = st.selectbox(
                "3D Scenario",
                options=list(SCENARIO_REGISTRY.keys()),
                format_func=lambda k: f"{SCENARIO_REGISTRY[k].title} ({getattr(SCENARIO_REGISTRY[k], 'isaac_task_id', k)})",
            )
        with c2:
            selected_prof = st.selectbox(
                "World Model Profile",
                options=list(HALLUCINATION_PROFILES.keys()),
                format_func=lambda k: HALLUCINATION_PROFILES[k]["label"],
            )
        with c3:
            run_live = st.button("🚀 Run Live Nemotron Audit", type="primary", use_container_width=True)
        with c4:
            run_bench = st.button("🔄 Audit All Scenarios", use_container_width=True)

        if run_live:
            with st.spinner(f"Auditing 3D scenario '{selected_scen}' ({selected_prof}) via Nebius Token Factory & Tavily..."):
                run_scenario_pipeline(scenario=selected_scen, profile=selected_prof, data_dir=DATA_DIR)
                all_reports = _load_all_reports()
            st.success(f"Completed 3D Physics Audit for `{selected_scen}`!")

        if run_bench:
            with st.spinner("Running full 3D benchmark across all scenarios..."):
                run_full_benchmark(profile=selected_prof, data_dir=DATA_DIR)
                all_reports = _load_all_reports()
            st.success("Completed full 3D Physics Audit benchmark!")

        rep = all_reports.get(selected_scen, next(iter(all_reports.values())))
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Audit Verdict", rep["overall_verdict"])
        k2.metric("Physics Hallucination Rate (PHR)", f"{rep['physics_hallucination_rate']:.3f}")
        k3.metric("Physical Integrity Rating (PIR)", f"{rep['physical_integrity_rating']:.3f}")
        k4.metric("SIPA Kinetic Debt", f"{rep['kinetic_debt_joules']:.1f} J")

        if rep.get("violations"):
            st.dataframe(pd.DataFrame(rep["violations"]), use_container_width=True, hide_index=True)

        franka_gt_dir = DATA_DIR / "ground_truth" / "franka_lift" / "frames"
        franka_gt_meta = DATA_DIR / "ground_truth" / "franka_lift" / "metadata.json"
        cosmos_pred_json = DATA_DIR / "predictions" / "franka_lift" / "scene_reasoning.json"
        franka_audit_json = DATA_DIR / "reports" / "franka_lift_audit.json"

        if franka_gt_dir.exists():
            st.markdown("#### 🤖 `Isaac-Lift-Cube-Franka-v0` Captured 3D Keyframes (`approach` → `pre-grasp` → `grasp` → `lift` → `hold`)")
            fcols = st.columns(5)
            stage_samples = [
                ("frame_00.png", "Step 00: Approach"),
                ("frame_06.png", "Step 06: Pre-Grasp"),
                ("frame_11.png", "Step 11: Grasp"),
                ("frame_17.png", "Step 17: Lift"),
                ("frame_23.png", "Step 23: Hold"),
            ]
            for col, (fname, caption) in zip(fcols, stage_samples):
                fpath = franka_gt_dir / fname
                if fpath.exists():
                    col.image(str(fpath), caption=caption, use_container_width=True)

        if cosmos_pred_json.exists() and franka_gt_meta.exists():
            st.markdown("#### 🌌 Step 3 & Step 4: `Isaac Lab → Cosmos3-Super-Reasoner → Nemotron-3-Nano-30B-A3B` Plan Audit")
            if st.button("⚖️ Re-Run Step 4 Plan Audit (Isaac Lab → Cosmos → Nemotron + Tavily)", key="btn_step4_plan_audit"):
                from auditor.nemotron_client import NemotronPhysicsAuditor

                with st.spinner("Auditing Cosmos manipulation plan with nvidia/Nemotron-3-Nano-30B-A3B on Nebius Token Factory..."):
                    aud = NemotronPhysicsAuditor()
                    aud.audit_scene_reasoning(
                        ground_truth_meta_path=franka_gt_meta,
                        scene_reasoning_path=cosmos_pred_json,
                        output_report_path=franka_audit_json,
                    )
                st.success("Updated `data/reports/franka_lift_audit.json` via Nebius Token Factory!")

            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.markdown("**Cosmos Prediction (`data/predictions/franka_lift/scene_reasoning.json`)**")
                st.code(cosmos_pred_json.read_text(encoding="utf-8"), language="json")
            with pcol2:
                st.markdown("**Nemotron Audit Report (`data/reports/franka_lift_audit.json`)**")
                if franka_audit_json.exists():
                    st.code(franka_audit_json.read_text(encoding="utf-8"), language="json")


if __name__ == "__main__":
    main()
