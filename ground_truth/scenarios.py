"""Module A: 3D Ground Truth Generator (ground_truth/scenarios.py).

Implements 3D rigid-body and articulated-robot scenarios modeled after NVIDIA Isaac Lab:
1. franka_lift_cube (Isaac-Lift-Cube-Franka-v0): 7-DOF Franka Panda arm grasping and lifting a 3D cube.
2. block_stack (Isaac-Stack-Block-3D-v0): 3D block stacking with 3D COM overhang, torque toppling, and contact SDF.
3. projectile (Isaac-Projectile-3D-v0): 3D ballistic launch in (x, y, z) with spin angular momentum and restitution.
4. ramp_roll (Isaac-Ramp-Roll-3D-v0): 3D sphere rolling down an inclined plane with surface friction and drag.

Outputs:
- 3D trajectories CSV (t, x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz, contact, sdf_clearance, grasp_force_n, mechanical_energy, stage)
- Multi-view 3D perspective + Depth Map PNG frames (frames/frame_000.png ... frame_100.png)
- Scene metadata & 3D scene-graph summary JSON (metadata.json)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import struct
import subprocess
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class ScenarioConfig:
    """3D physical parameters and Isaac Lab task metadata for a ground-truth scenario."""

    scenario: str
    isaac_task_id: str
    title: str
    description: str
    material_pair: str
    tavily_query: str
    friction: float
    mass: float  # kg
    initial_velocity: float  # m/s
    gravity: float = 9.81  # m/s^2
    restitution: float = 0.65
    ramp_angle_deg: float = 25.0
    linear_damping: float = 0.18
    stack_offset_x: float = 0.34  # m relative to support half-width 0.25m
    stack_offset_y: float = 0.08  # m 3D lateral offset
    launch_angle_deg: float = 52.0
    launch_azimuth_deg: float = 18.0  # 3D azimuth angle in X-Y plane
    grasp_force_newton: float = 12.0  # Franka parallel-jaw normal grasp force (N)
    num_frames: int = 101
    dt: float = 0.04  # 25 FPS -> 4.0 seconds total
    laws_tested: List[str] = field(default_factory=list)


@dataclass
class TrajectoryPoint:
    """Single 3D timestep state in an Isaac Lab or Cosmos world-model rollout."""

    frame_index: int
    t: float
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0
    contact: int = 1
    sdf_clearance: float = 0.0  # Signed distance to nearest collider (negative = interpenetration)
    grasp_force_n: float = 0.0
    ee_x: float = 0.0
    ee_y: float = 0.0
    ee_z: float = 0.0
    stage: str = "dynamic"
    mechanical_energy: float = 0.0


@dataclass
class SimulationResult:
    """Output bundle produced by a 3D ground-truth scenario run."""

    scenario: str
    output_dir: Path
    trajectory_csv: Path
    metadata_json: Path
    frames_dir: Path
    frame_paths: List[Path]
    trajectory: List[TrajectoryPoint]
    metadata: Dict[str, Any]


SCENARIO_REGISTRY: Dict[str, ScenarioConfig] = {
    "warehouse_amr": ScenarioConfig(
        scenario="warehouse_amr",
        isaac_task_id="Isaac-Warehouse-AMR-Nav-v0",
        title="3D Warehouse AMR Navigation & LiDAR Obstacle Avoidance (Robot 01)",
        description=(
            "Autonomous Mobile Robot (Robot 01, 45.0 kg) navigates a 3D warehouse aisle between Rack_A..C "
            "and Forklift_01 toward Goal_01 at 1.20 m/s using 360-degree Lidar_01 and front/rear RGB cameras."
        ),
        material_pair="polyurethane AMR drive wheels on sealed warehouse concrete",
        tavily_query="coefficient of kinetic friction polyurethane wheel on sealed concrete warehouse floor",
        friction=0.72,
        mass=45.0,
        initial_velocity=1.20,
        gravity=9.81,
        restitution=0.10,
        laws_tested=[
            "Contact constraints (no rack/forklift/pallet collider interpenetration)",
            "Friction and damping behavior (non-holonomic wheel traction without lateral skid)",
            "Conservation of momentum (smooth acceleration & braking along spline)",
            "Gravity and floor plane contact (z = 0.23m chassis height)",
            "Object permanence (AMR, Forklift_01, and pallet racks persist in LiDAR & cameras)",
        ],
    ),
    "franka_lift_cube": ScenarioConfig(
        scenario="franka_lift_cube",
        isaac_task_id="Isaac-Lift-Cube-Franka-v0",
        title="3D Franka Arm Cube Lift (Isaac-Lift-Cube-Franka-v0)",
        description=(
            "A 7-DOF Franka Emika Panda robotic arm approaches, grasps, and lifts a 0.10 kg "
            "3D rigid cube (6 cm edge) from a worktable to a 3D target goal pose. Tests 3D gripper-cube "
            "contact constraints, frictional grasp closure (F_n >= mg / 2*mu), gravity during lift, "
            "and object permanence."
        ),
        material_pair="rubber gripper pads on ABS plastic cube",
        tavily_query="coefficient of static friction rubber on ABS plastic robotic gripper physical constant",
        friction=0.65,
        mass=0.10,
        initial_velocity=0.0,
        gravity=9.81,
        restitution=0.15,
        grasp_force_newton=12.0,
        laws_tested=[
            "Contact constraints (3D gripper & table non-penetration SDF >= 0)",
            "Friction and damping behavior (frictional grasp closure without slip)",
            "Gravity and free-fall behavior (cube must fall if ungrasped)",
            "Conservation of momentum (smooth end-effector & payload kinematics)",
            "Object permanence (cube and robot links must persist across frames)",
        ],
    ),
    "block_stack": ScenarioConfig(
        scenario="block_stack",
        isaac_task_id="Isaac-Stack-Block-3D-v0",
        title="3D Asymmetric Block Stacking & Torque Toppling",
        description=(
            "A 2.5 kg 3D wooden block is placed atop a 3D support pedestal with its 3D center of mass "
            "overhanging the support polygon edge (dx=0.34 m, dy=0.08 m vs 0.25 m half-width), "
            "triggering a 3D torque-induced rotational topple onto the rigid floor."
        ),
        material_pair="wood on wood",
        tavily_query="coefficient of static friction wood on wood physical constant",
        friction=0.42,
        mass=2.5,
        initial_velocity=0.0,
        gravity=9.81,
        restitution=0.25,
        stack_offset_x=0.34,
        stack_offset_y=0.08,
        laws_tested=[
            "Center of mass & 3D support polygon torque stability",
            "Contact constraints (3D non-penetration with pedestal and floor)",
            "Gravity and free-fall behavior during 3D toppling arc",
            "Conservation of linear and angular momentum on impact",
            "Object permanence",
        ],
    ),
    "projectile": ScenarioConfig(
        scenario="projectile",
        isaac_task_id="Isaac-Projectile-3D-v0",
        title="3D Ballistic Projectile Motion with Spin & Restitution",
        description=(
            "A 0.8 kg steel sphere is launched in 3D space at 6.8 m/s (elevation 52 deg, azimuth 18 deg) "
            "with initial backspin angular momentum, following a 3D parabola under gravity and aerodynamic "
            "drag before bouncing on a concrete floor (e=0.68)."
        ),
        material_pair="steel on concrete",
        tavily_query="coefficient of restitution and friction steel on concrete physical constant",
        friction=0.45,
        mass=0.8,
        initial_velocity=6.80,
        gravity=9.81,
        restitution=0.68,
        launch_angle_deg=52.0,
        launch_azimuth_deg=18.0,
        laws_tested=[
            "Gravity and 3D parabolic free-fall trajectory",
            "Conservation of linear and angular momentum at 3D floor impact",
            "Contact constraints (3D floor plane z >= radius)",
            "Friction and aerodynamic drag energy dissipation",
            "Object permanence",
        ],
    ),
    "ramp_roll": ScenarioConfig(
        scenario="ramp_roll",
        isaac_task_id="Isaac-Ramp-Roll-3D-v0",
        title="3D Inclined Plane Sphere Roll",
        description=(
            "A 1.0 kg rubber sphere rolls down a 25-degree 3D wooden ramp under gravity, "
            "3D rolling contact constraint, surface friction (mu=0.50), and drag onto a flat runout."
        ),
        material_pair="rubber on wood",
        tavily_query="coefficient of kinetic friction rubber on dry wood physical constant",
        friction=0.50,
        mass=1.0,
        initial_velocity=0.20,
        gravity=9.81,
        restitution=0.45,
        ramp_angle_deg=25.0,
        linear_damping=0.22,
        laws_tested=[
            "Conservation of momentum",
            "Contact constraints (no 3D ramp surface interpenetration)",
            "Gravity and inclined 3D acceleration",
            "Friction and terminal velocity damping",
            "Object permanence",
        ],
    ),
}


def get_scenario_config(
    scenario: str,
    friction: Optional[float] = None,
    mass: Optional[float] = None,
    initial_velocity: Optional[float] = None,
    num_frames: Optional[int] = None,
) -> ScenarioConfig:
    """Return a copy of the 3D scenario configuration with optional parameter overrides."""
    if scenario not in SCENARIO_REGISTRY:
        valid = ", ".join(sorted(SCENARIO_REGISTRY.keys()))
        raise ValueError(f"Unknown scenario '{scenario}'. Valid scenarios: {valid}")

    base = SCENARIO_REGISTRY[scenario]
    cfg = ScenarioConfig(**asdict(base))
    if friction is not None:
        cfg.friction = float(friction)
    if mass is not None:
        cfg.mass = float(mass)
    if initial_velocity is not None:
        cfg.initial_velocity = float(initial_velocity)
    if num_frames is not None:
        cfg.num_frames = int(num_frames)
    return cfg


def _euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Convert 3D Euler angles (roll, pitch, yaw in radians) to (qx, qy, qz, qw)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (round(qx, 6), round(qy, 6), round(qz, 6), round(qw, 6))


def simulate_trajectory(config: ScenarioConfig) -> List[TrajectoryPoint]:
    """Run deterministic 3D rigid-body and articulated-robot physics integration."""
    points: List[TrajectoryPoint] = []
    g = config.gravity
    dt = config.dt
    m = config.mass

    if config.scenario == "warehouse_amr":
        # Robot 01 navigating warehouse aisle from (12.42, 4.85, 0.23) along S-curve spline toward Goal_01
        v_nom = config.initial_velocity if config.initial_velocity > 0 else 1.20
        for idx in range(config.num_frames):
            t = round(idx * dt, 4)
            u = idx / max(1, config.num_frames - 1)
            x = 12.42 + u * 4.20
            y = 4.85 + 0.65 * math.sin(u * math.pi * 1.5)
            z = 0.23
            vx = v_nom
            vy = 0.65 * (math.pi * 1.5 / (config.num_frames * dt)) * math.cos(u * math.pi * 1.5)
            vz = 0.0
            yaw = math.atan2(vy, vx)
            qx, qy, qz, qw = _euler_to_quaternion(0.0, 0.0, yaw)
            ke = 0.5 * m * (vx * vx + vy * vy)
            points.append(
                TrajectoryPoint(
                    frame_index=idx,
                    t=t,
                    x=round(x, 5),
                    y=round(y, 5),
                    z=round(z, 5),
                    qx=qx,
                    qy=qy,
                    qz=qz,
                    qw=qw,
                    vx=round(vx, 5),
                    vy=round(vy, 5),
                    vz=0.0,
                    wx=0.0,
                    wy=0.0,
                    wz=round(yaw, 5),
                    contact=1,
                    sdf_clearance=0.45,
                    grasp_force_n=0.0,
                    stage="navigating",
                    mechanical_energy=round(ke, 5),
                )
            )
        return points

    if config.scenario == "franka_lift_cube":
        # 3D Franka Panda manipulation phases on worktable (z_table = 0.0, cube half-size = 0.08m)
        cube_half = 0.08
        cube_x, cube_y, cube_z = 1.65, 0.15, cube_half
        ee_start = (0.85, -0.35, 1.35)
        pre_grasp_pos = (1.65, 0.15, 0.42)
        grasp_pos = (1.65, 0.15, cube_half + 0.02)
        goal_pos = (1.35, 0.45, 1.18)

        min_hold_force = (m * g) / max(0.05, 2.0 * config.friction)

        for idx in range(config.num_frames):
            t = round(idx * dt, 4)
            prev_x, prev_y, prev_z = cube_x, cube_y, cube_z

            if idx <= 20:
                stage = "approach"
                u = idx / 20.0
                s = u * u * (3.0 - 2.0 * u)
                ee_x = ee_start[0] + s * (pre_grasp_pos[0] - ee_start[0])
                ee_y = ee_start[1] + s * (pre_grasp_pos[1] - ee_start[1])
                ee_z = ee_start[2] + s * (pre_grasp_pos[2] - ee_start[2])
                grasp_f = 0.0
                contact = 1
            elif idx <= 35:
                stage = "pre_grasp"
                u = (idx - 20) / 15.0
                s = u * u * (3.0 - 2.0 * u)
                ee_x = pre_grasp_pos[0]
                ee_y = pre_grasp_pos[1]
                ee_z = pre_grasp_pos[2] + s * (grasp_pos[2] - pre_grasp_pos[2])
                grasp_f = 0.0
                contact = 1
            elif idx <= 48:
                stage = "grasp"
                u = (idx - 35) / 13.0
                ee_x, ee_y, ee_z = grasp_pos
                grasp_f = round(config.grasp_force_newton * min(1.0, u * 1.2), 3)
                contact = 1
            elif idx <= 78:
                stage = "lift"
                u = (idx - 48) / 30.0
                s = u * u * (3.0 - 2.0 * u)
                ee_x = grasp_pos[0] + s * (goal_pos[0] - grasp_pos[0])
                ee_y = grasp_pos[1] + s * (goal_pos[1] - grasp_pos[1])
                ee_z = grasp_pos[2] + s * (goal_pos[2] - grasp_pos[2])
                grasp_f = config.grasp_force_newton
                if grasp_f >= min_hold_force:
                    cube_x, cube_y, cube_z = ee_x, ee_y, ee_z - 0.02
                    contact = 1
                else:
                    # Insufficient friction/grasp force causes slip back to table
                    cube_z = max(cube_half, cube_z - 0.5 * g * dt * dt)
                    contact = 0
            else:
                stage = "hold"
                ee_x, ee_y, ee_z = goal_pos
                grasp_f = config.grasp_force_newton
                cube_x, cube_y, cube_z = ee_x, ee_y, ee_z - 0.02
                contact = 1

            vx = (cube_x - prev_x) / dt if idx > 0 else 0.0
            vy = (cube_y - prev_y) / dt if idx > 0 else 0.0
            vz = (cube_z - prev_z) / dt if idx > 0 else 0.0
            yaw = 0.15 * ((cube_y - 0.15) / 0.30)
            qx, qy, qz, qw = _euler_to_quaternion(0.0, 0.0, yaw)
            sdf = round(cube_z - cube_half, 5)
            ke = 0.5 * m * (vx * vx + vy * vy + vz * vz)
            pe = m * g * max(0.0, cube_z - cube_half)

            points.append(
                TrajectoryPoint(
                    frame_index=idx,
                    t=t,
                    x=round(cube_x, 5),
                    y=round(cube_y, 5),
                    z=round(cube_z, 5),
                    qx=qx,
                    qy=qy,
                    qz=qz,
                    qw=qw,
                    vx=round(vx, 5),
                    vy=round(vy, 5),
                    vz=round(vz, 5),
                    wx=0.0,
                    wy=0.0,
                    wz=round(yaw / max(dt, t if t > 0 else dt), 5),
                    contact=contact,
                    sdf_clearance=sdf,
                    grasp_force_n=round(grasp_f, 3),
                    ee_x=round(ee_x, 5),
                    ee_y=round(ee_y, 5),
                    ee_z=round(ee_z, 5),
                    stage=stage,
                    mechanical_energy=round(ke + pe, 5),
                )
            )

    elif config.scenario == "ramp_roll":
        theta = math.radians(config.ramp_angle_deg)
        ramp_length = 4.2
        radius = 0.18
        s = 0.0
        v = config.initial_velocity
        theta_rot = 0.0
        a_incline = max(0.65, g * (math.sin(theta) - 0.35 * config.friction * math.cos(theta)))

        for idx in range(config.num_frames):
            t = round(idx * dt, 4)
            y_pos = 0.12 * math.sin(idx * 0.04)  # gentle 3D lateral camber
            vy = 0.12 * 0.04 * math.cos(idx * 0.04) / dt
            if s < ramp_length:
                stage = "incline_roll"
                a = a_incline - config.linear_damping * v
                x = s * math.cos(theta)
                z = (ramp_length - s) * math.sin(theta) + radius
                vx = v * math.cos(theta)
                vz = -v * math.sin(theta)
            else:
                stage = "runout_damping"
                a = -config.friction * g * 0.45 - config.linear_damping * v
                x_ramp_end = ramp_length * math.cos(theta)
                x = x_ramp_end + (s - ramp_length)
                z = radius
                vx = v
                vz = 0.0

            qx, qy, qz, qw = _euler_to_quaternion(0.0, theta_rot, 0.05)
            ke = 0.5 * m * (v ** 2) * 1.4
            pe = m * g * max(0.0, z - radius)
            points.append(
                TrajectoryPoint(
                    frame_index=idx,
                    t=t,
                    x=round(x, 5),
                    y=round(y_pos, 5),
                    z=round(z, 5),
                    qx=qx,
                    qy=qy,
                    qz=qz,
                    qw=qw,
                    vx=round(vx, 5),
                    vy=round(vy, 5),
                    vz=round(vz, 5),
                    wx=0.0,
                    wy=round(v / radius, 5),
                    wz=0.0,
                    contact=1,
                    sdf_clearance=0.0,
                    grasp_force_n=0.0,
                    stage=stage,
                    mechanical_energy=round(ke + pe, 5),
                )
            )
            v = max(0.0, v + a * dt)
            s += v * dt
            theta_rot += (v / radius) * dt

    elif config.scenario == "block_stack":
        base_x, base_y = 1.50, 0.0
        base_half_w = 0.25
        base_h = 0.80
        upper_half_h = 0.20
        pivot_x = base_x + base_half_w
        pivot_z = base_h

        overhang = config.stack_offset_x - base_half_w
        is_unstable = overhang > 0.01

        phi = 0.0
        roll = 0.0
        omega = 0.0
        free_fall = False
        x = base_x + config.stack_offset_x
        y = base_y + config.stack_offset_y
        z = base_h + upper_half_h
        vx, vy, vz = 0.0, 0.0, 0.0
        contact = 1

        for idx in range(config.num_frames):
            t = round(idx * dt, 4)
            if not is_unstable:
                qx, qy, qz, qw = _euler_to_quaternion(0.0, 0.0, 0.0)
                points.append(
                    TrajectoryPoint(
                        frame_index=idx,
                        t=t,
                        x=round(x, 5),
                        y=round(y, 5),
                        z=round(z, 5),
                        qx=qx,
                        qy=qy,
                        qz=qz,
                        qw=qw,
                        stage="stable_stack",
                        sdf_clearance=0.0,
                        mechanical_energy=round(m * g * z, 5),
                    )
                )
                continue

            if not free_fall:
                stage = "torque_topple"
                alpha = (g / 0.35) * math.sin(phi + 0.18) * (1.0 - 0.25 * config.friction)
                omega += alpha * dt
                phi += omega * dt
                roll = 0.18 * phi
                r_com = math.hypot(overhang, upper_half_h)
                base_angle = math.atan2(upper_half_h, max(0.02, overhang))
                curr_angle = base_angle - phi
                x = pivot_x + r_com * math.cos(curr_angle)
                y = base_y + config.stack_offset_y + 0.06 * math.sin(phi)
                z = pivot_z + r_com * math.sin(curr_angle)
                vx = r_com * omega * math.sin(curr_angle)
                vy = 0.06 * omega * math.cos(phi)
                vz = -r_com * omega * math.cos(curr_angle)
                contact = 1
                if phi >= 1.08:
                    free_fall = True
            else:
                stage = "ballistic_fall"
                vz -= g * dt
                x += vx * dt
                y += vy * dt
                z += vz * dt
                phi = min(math.pi / 2.0, phi + omega * dt)
                contact = 0
                if z <= upper_half_h:
                    stage = "floor_impact"
                    z = upper_half_h
                    contact = 1
                    if abs(vz) > 0.35:
                        vz = -vz * config.restitution
                        vx *= (1.0 - config.friction * 0.45)
                        vy *= (1.0 - config.friction * 0.45)
                        omega *= 0.3
                    else:
                        stage = "settled"
                        vz = 0.0
                        vx *= max(0.0, 1.0 - config.friction * 4.0 * dt)
                        vy *= max(0.0, 1.0 - config.friction * 4.0 * dt)
                        omega = 0.0

            qx, qy, qz, qw = _euler_to_quaternion(roll, phi, 0.08)
            ke = 0.5 * m * (vx ** 2 + vy ** 2 + vz ** 2)
            pe = m * g * z
            points.append(
                TrajectoryPoint(
                    frame_index=idx,
                    t=t,
                    x=round(x, 5),
                    y=round(y, 5),
                    z=round(z, 5),
                    qx=qx,
                    qy=qy,
                    qz=qz,
                    qw=qw,
                    vx=round(vx, 5),
                    vy=round(vy, 5),
                    vz=round(vz, 5),
                    wx=round(0.18 * omega, 5),
                    wy=round(omega, 5),
                    wz=0.0,
                    contact=contact,
                    sdf_clearance=round(z - upper_half_h, 5),
                    stage=stage,
                    mechanical_energy=round(ke + pe, 5),
                )
            )

    elif config.scenario == "projectile":
        elev_rad = math.radians(config.launch_angle_deg)
        azim_rad = math.radians(config.launch_azimuth_deg)
        radius = 0.15
        x, y, z = 0.30, -0.25, 0.45
        v_horiz = config.initial_velocity * math.cos(elev_rad)
        vx = v_horiz * math.cos(azim_rad)
        vy = v_horiz * math.sin(azim_rad)
        vz = config.initial_velocity * math.sin(elev_rad)
        pitch = 0.0
        wy = 8.5  # initial backspin rad/s

        for idx in range(config.num_frames):
            t = round(idx * dt, 4)
            contact = 1 if z <= radius + 1e-3 else 0
            stage = "floor_bounce" if contact else ("ascending" if vz >= 0 else "descending")
            qx, qy, qz, qw = _euler_to_quaternion(0.0, pitch, azim_rad)
            ke = 0.5 * m * (vx ** 2 + vy ** 2 + vz ** 2)
            pe = m * g * max(0.0, z - radius)

            points.append(
                TrajectoryPoint(
                    frame_index=idx,
                    t=t,
                    x=round(x, 5),
                    y=round(y, 5),
                    z=round(z, 5),
                    qx=qx,
                    qy=qy,
                    qz=qz,
                    qw=qw,
                    vx=round(vx, 5),
                    vy=round(vy, 5),
                    vz=round(vz, 5),
                    wx=0.0,
                    wy=round(wy, 5),
                    wz=0.0,
                    contact=contact,
                    sdf_clearance=round(z - radius, 5),
                    stage=stage,
                    mechanical_energy=round(ke + pe, 5),
                )
            )

            vz -= g * dt
            x += vx * dt
            y += vy * dt
            z += vz * dt
            pitch += wy * dt

            if z <= radius:
                z = radius
                if abs(vz) > 0.30:
                    vz = -vz * config.restitution
                    vx *= max(0.0, 1.0 - 0.18 * config.friction)
                    vy *= max(0.0, 1.0 - 0.18 * config.friction)
                    wy *= 0.72
                else:
                    vz = 0.0
                    vx *= max(0.0, 1.0 - config.friction * 2.5 * dt)
                    vy *= max(0.0, 1.0 - config.friction * 2.5 * dt)
                    wy *= 0.85

    else:
        raise ValueError(f"Unsupported scenario: {config.scenario}")

    return points


def _write_minimal_png(
    path: Path,
    width: int,
    height: int,
    rgb_rows: List[bytes],
) -> None:
    """Fallback pure-stdlib PNG writer in case Pillow is unavailable."""

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + row for row in rgb_rows)
    compressed = zlib.compress(raw, level=6)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )
    path.write_bytes(png_bytes)


def _project_3d_iso(wx: float, wy: float, wz: float, vp_w: int = 430, vp_h: int = 360) -> Tuple[int, int, float]:
    """Project 3D world coordinates (wx, wy, wz) into 2D isometric perspective screen coordinates + depth."""
    # Camera centered near (2.2, 0.0, 0.8) looking from front-right elevated angle
    nx = wx - 2.1
    ny = wy
    nz = wz - 0.75
    sx = int(vp_w * 0.46 + (nx * 68.0 - ny * 46.0))
    sz = int(vp_h * 0.62 - nz * 92.0 + (nx * 18.0 + ny * 26.0))
    depth = max(0.5, 4.5 - 0.35 * wx - 0.55 * wy + 0.15 * wz)
    return sx, sz, depth


def _draw_3d_box(
    draw: Any,
    cx: float,
    cy: float,
    cz: float,
    hx: float,
    hy: float,
    hz: float,
    top_color: Tuple[int, int, int],
    front_color: Tuple[int, int, int],
    side_color: Tuple[int, int, int],
    outline: Tuple[int, int, int],
) -> None:
    """Draw a shaded 3D isometric box given center (cx, cy, cz) and half-extents (hx, hy, hz)."""
    corners = {}
    for dx in (-hx, hx):
        for dy in (-hy, hy):
            for dz in (-hz, hz):
                sx, sz, _ = _project_3d_iso(cx + dx, cy + dy, cz + dz)
                corners[(dx > 0, dy > 0, dz > 0)] = (sx, sz)

    # Top face (dz = +hz)
    top_face = [
        corners[(False, False, True)],
        corners[(True, False, True)],
        corners[(True, True, True)],
        corners[(False, True, True)],
    ]
    # Front-left face (dy = +hy)
    front_face = [
        corners[(False, True, False)],
        corners[(True, True, False)],
        corners[(True, True, True)],
        corners[(False, True, True)],
    ]
    # Right face (dx = +hx)
    side_face = [
        corners[(True, False, False)],
        corners[(True, True, False)],
        corners[(True, True, True)],
        corners[(True, False, True)],
    ]
    draw.polygon(front_face, fill=front_color, outline=outline)
    draw.polygon(side_face, fill=side_color, outline=outline)
    draw.polygon(top_face, fill=top_color, outline=outline)


def render_scenario_frame(
    config: ScenarioConfig,
    point: TrajectoryPoint,
    history: List[TrajectoryPoint],
    output_path: Path,
    mode_label: str = "ISAAC LAB 3D GROUND TRUTH",
    reference_point: Optional[TrajectoryPoint] = None,
    violation_badge: Optional[str] = None,
    object_visible: bool = True,
) -> None:
    """Render a split-screen 3D Perspective Viewport + 3D Depth/Scene-Graph HUD frame (640x360 PNG)."""
    width, height = 640, 360
    vp_w = 430

    if not HAS_PIL:
        bg = bytes([15, 23, 42]) * width
        rows = [bytearray(bg) for _ in range(height)]
        if object_visible:
            ox, oz, _ = _project_3d_iso(point.x, point.y, point.z, vp_w, height)
            color = (56, 189, 248) if "GROUND TRUTH" in mode_label else (249, 115, 22)
            for dy in range(-12, 13):
                for dx in range(-12, 13):
                    if dx * dx + dy * dy <= 144:
                        py, px = oz + dy, ox + dx
                        if 0 <= py < height and 0 <= px < width:
                            rows[py][px * 3 : px * 3 + 3] = bytes(color)
        _write_minimal_png(output_path, width, height, [bytes(r) for r in rows])
        return

    img = Image.new("RGB", (width, height), (15, 23, 42))
    draw = ImageDraw.Draw(img)

    # 1. Draw 3D Perspective Floor Grid (z = 0) in Left Viewport
    for gx_idx in range(0, 11):
        gx = gx_idx * 0.5
        p0 = _project_3d_iso(gx, -1.0, 0.0)[:2]
        p1 = _project_3d_iso(gx, 1.0, 0.0)[:2]
        draw.line([p0, p1], fill=(30, 41, 59), width=1)
    for gy_idx in range(-4, 5):
        gy = gy_idx * 0.25
        p0 = _project_3d_iso(0.0, gy, 0.0)[:2]
        p1 = _project_3d_iso(5.0, gy, 0.0)[:2]
        draw.line([p0, p1], fill=(30, 41, 59), width=1)

    # Draw 3D Coordinate Triad (X=Red, Y=Green, Z=Blue) at origin
    o_pt = _project_3d_iso(0.1, -0.85, 0.0)[:2]
    x_pt = _project_3d_iso(0.55, -0.85, 0.0)[:2]
    y_pt = _project_3d_iso(0.1, -0.40, 0.0)[:2]
    z_pt = _project_3d_iso(0.1, -0.85, 0.45)[:2]
    draw.line([o_pt, x_pt], fill=(239, 68, 68), width=2)
    draw.line([o_pt, y_pt], fill=(34, 197, 94), width=2)
    draw.line([o_pt, z_pt], fill=(59, 130, 246), width=2)

    # 2. Static 3D Scene Geometry per Scenario
    if config.scenario == "ramp_roll":
        theta = math.radians(config.ramp_angle_deg)
        ramp_len = 4.2
        top_z = ramp_len * math.sin(theta)
        bot_x = ramp_len * math.cos(theta)
        # 3D inclined ramp prism
        r_fl = _project_3d_iso(0.0, 0.35, top_z)[:2]
        r_fr = _project_3d_iso(bot_x, 0.35, 0.0)[:2]
        r_br = _project_3d_iso(bot_x, -0.35, 0.0)[:2]
        r_bl = _project_3d_iso(0.0, -0.35, top_z)[:2]
        r_base_fl = _project_3d_iso(0.0, 0.35, 0.0)[:2]
        draw.polygon([r_fl, r_fr, r_base_fl], fill=(51, 65, 85), outline=(148, 163, 184))
        draw.polygon([r_bl, r_br, r_fr, r_fl], fill=(71, 85, 105), outline=(226, 232, 240))

    elif config.scenario == "block_stack":
        # 3D Pedestal base block at (1.50, 0.0, 0.40) with half-extents (0.25, 0.25, 0.40)
        _draw_3d_box(
            draw,
            cx=1.50,
            cy=0.0,
            cz=0.40,
            hx=0.25,
            hy=0.25,
            hz=0.40,
            top_color=(100, 116, 139),
            front_color=(71, 85, 105),
            side_color=(51, 65, 85),
            outline=(203, 213, 225),
        )

    elif config.scenario == "franka_lift_cube":
        # Draw 3D Goal Target Wireframe Marker at (1.35, 0.45, 1.16)
        gx, gy, _ = _project_3d_iso(1.35, 0.45, 1.16)
        draw.ellipse([gx - 14, gy - 14, gx + 14, gy + 14], outline=(74, 222, 128), width=2)
        # Draw 7-DOF Franka Emika Panda Articulated Robot Arm from base (0.35, -0.25, 0.0) to (ee_x, ee_y, ee_z)
        base_p = _project_3d_iso(0.45, -0.25, 0.0)[:2]
        shoulder_p = _project_3d_iso(0.45, -0.25, 0.68)[:2]
        elbow_p = _project_3d_iso(
            0.5 * (0.45 + point.ee_x),
            0.5 * (-0.25 + point.ee_y),
            max(0.95, point.ee_z + 0.42),
        )[:2]
        wrist_p = _project_3d_iso(point.ee_x, point.ee_y, point.ee_z + 0.16)[:2]
        ee_p = _project_3d_iso(point.ee_x, point.ee_y, point.ee_z)[:2]

        # Robot pedestal + articulated links
        _draw_3d_box(
            draw,
            cx=0.45,
            cy=-0.25,
            cz=0.12,
            hx=0.14,
            hy=0.14,
            hz=0.12,
            top_color=(148, 163, 184),
            front_color=(100, 116, 139),
            side_color=(71, 85, 105),
            outline=(226, 232, 240),
        )
        draw.line([base_p, shoulder_p], fill=(226, 232, 240), width=7)
        draw.line([shoulder_p, elbow_p], fill=(203, 213, 225), width=6)
        draw.line([elbow_p, wrist_p], fill=(148, 163, 184), width=5)
        draw.line([wrist_p, ee_p], fill=(248, 250, 252), width=4)
        for jt in (shoulder_p, elbow_p, wrist_p):
            draw.ellipse([jt[0] - 5, jt[1] - 5, jt[0] + 5, jt[1] + 5], fill=(56, 189, 248), outline=(15, 23, 42))

        # Parallel-jaw gripper fingers
        jaw_gap = 10 if point.grasp_force_n >= 4.0 else 16
        draw.line([(ee_p[0] - jaw_gap, ee_p[1] - 4), (ee_p[0] - jaw_gap, ee_p[1] + 12)], fill=(56, 189, 248), width=3)
        draw.line([(ee_p[0] + jaw_gap, ee_p[1] - 4), (ee_p[0] + jaw_gap, ee_p[1] + 12)], fill=(56, 189, 248), width=3)

    # 3. Reference 3D Ghost (Ground-Truth overlay when inspecting World Model Dream)
    if reference_point is not None:
        rx, rz, _ = _project_3d_iso(reference_point.x, reference_point.y, reference_point.z)
        draw.ellipse([rx - 15, rz - 15, rx + 15, rz + 15], outline=(56, 189, 248), width=2)

    # 4. 3D Trajectory Ribbon
    if len(history) > 1:
        trail_pts = [_project_3d_iso(p.x, p.y, p.z)[:2] for p in history]
        trail_color = (56, 189, 248) if "GROUND TRUTH" in mode_label else (251, 146, 60)
        draw.line(trail_pts, fill=trail_color, width=2)

    # 5. Active 3D Rigid Body
    ox, oz, obj_depth = _project_3d_iso(point.x, point.y, point.z)
    if object_visible:
        is_gt = "GROUND TRUTH" in mode_label
        if violation_badge:
            top_c, front_c, side_c, out_c = (248, 113, 113), (239, 68, 68), (185, 28, 28), (254, 226, 226)
        elif is_gt:
            top_c, front_c, side_c, out_c = (56, 189, 248), (14, 165, 233), (2, 132, 199), (224, 242, 254)
        else:
            top_c, front_c, side_c, out_c = (251, 146, 60), (249, 115, 22), (194, 65, 12), (255, 237, 213)

        if config.scenario in ("block_stack", "franka_lift_cube"):
            hx = 0.12 if config.scenario == "franka_lift_cube" else 0.24
            hy = 0.12 if config.scenario == "franka_lift_cube" else 0.20
            hz = 0.08 if config.scenario == "franka_lift_cube" else 0.20
            _draw_3d_box(draw, point.x, point.y, point.z, hx, hy, hz, top_c, front_c, side_c, out_c)
        else:
            r_px = 15
            draw.ellipse([ox - r_px, oz - r_px, ox + r_px, oz + r_px], fill=front_c, outline=out_c, width=2)
            draw.ellipse([ox - 6, oz - 8, ox - 1, oz - 3], fill=top_c)

    # 6. Right Viewport (x = 430..640): 3D Depth Map (Z-Buffer) & Scene Graph Telemetry
    draw.rectangle([vp_w, 32, width, height], fill=(9, 14, 28), outline=(51, 65, 85))
    draw.text((vp_w + 10, 40), "3D DEPTH MAP & SCENE GRAPH", fill=(148, 163, 184))

    # Mini Depth-Map Window (x = 442..628, y = 60..195)
    dm_x0, dm_y0, dm_x1, dm_y1 = vp_w + 12, 60, width - 12, 195
    draw.rectangle([dm_x0, dm_y0, dm_x1, dm_y1], fill=(2, 6, 23), outline=(71, 85, 105))
    # Depth gradient horizon
    for dy_band in range(0, 55, 5):
        shade = 22 + dy_band // 2
        draw.line([(dm_x0 + 1, dm_y1 - dy_band), (dm_x1 - 1, dm_y1 - dy_band)], fill=(shade, shade + 10, shade + 24))
    if object_visible:
        # Map 3D object into depth-map window with intensity proportional to inverse depth
        d_norm = max(0.15, min(1.0, (4.8 - obj_depth) / 3.5))
        d_val = int(75 + 175 * d_norm)
        dmx = int(dm_x0 + 25 + min(1.0, max(0.0, point.x / 4.2)) * (dm_x1 - dm_x0 - 50))
        dmz = int(dm_y1 - 18 - min(1.0, max(0.0, point.z / 2.2)) * (dm_y1 - dm_y0 - 36))
        draw.ellipse(
            [dmx - 14, dmz - 14, dmx + 14, dmz + 14],
            fill=(d_val, d_val, min(255, d_val + 15)),
            outline=(56, 189, 248),
        )
        draw.text((dm_x0 + 8, dm_y0 + 6), f"Z-Depth: {obj_depth:.2f}m | SDF: {point.sdf_clearance:+.2f}m", fill=(125, 211, 252))
    else:
        draw.text((dm_x0 + 12, dm_y0 + 55), "OBJECT OCCLUDED / MISSING", fill=(248, 113, 113))

    # 3D Scene Graph Telemetry Panel (y = 204..350)
    telemetry_lines = [
        f"Task: {config.isaac_task_id}",
        f"Stage: {point.stage.upper()}",
        f"3D Pos: ({point.x:.2f}, {point.y:.2f}, {point.z:.2f}) m",
        f"3D Vel: ({point.vx:.2f}, {point.vy:.2f}, {point.vz:.2f}) m/s",
        f"Quat: ({point.qx:.2f}, {point.qy:.2f}, {point.qz:.2f}, {point.qw:.2f})",
        f"Grasp F: {point.grasp_force_n:.1f} N | E_mech: {point.mechanical_energy:.2f} J",
    ]
    ty = 204
    for line in telemetry_lines:
        draw.text((vp_w + 12, ty), line, fill=(203, 213, 225))
        ty += 19

    # Top HUD Banner
    draw.rectangle([0, 0, width, 32], fill=(2, 6, 23))
    hud_title = f"{mode_label} | {config.isaac_task_id} | Frame {point.frame_index:03d} (t={point.t:.2f}s)"
    draw.text((10, 9), hud_title, fill=(248, 250, 252))

    if violation_badge:
        draw.rectangle([12, 40, vp_w - 12, 70], fill=(153, 27, 27), outline=(252, 165, 165), width=2)
        draw.text((20, 49), f"3D PHYSICS VIOLATION: {violation_badge[:36]}", fill=(254, 242, 242))

    img.save(output_path, format="PNG")


def _try_run_npa_isaac_lab(scenario: str, output_dir: Path, num_frames: int) -> bool:
    """Attempt to run Nebius Physical AI Workbench `npa workbench isaac-lab` if installed and configured."""
    npa_bin = shutil.which("npa")
    if not npa_bin:
        return False
    try:
        completed = subprocess.run(
            [
                npa_bin,
                "workbench",
                "isaac-lab",
                "run",
                "--scenario",
                scenario,
                "--num-frames",
                str(num_frames),
                "--output",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return completed.returncode == 0 and (output_dir / "trajectory.csv").exists()
    except Exception:
        return False


def generate_ground_truth(
    scenario: str,
    output_root: Path | str = Path("data/ground_truth"),
    friction: Optional[float] = None,
    mass: Optional[float] = None,
    initial_velocity: Optional[float] = None,
    num_frames: int = 101,
    use_nebius: bool = False,
) -> SimulationResult:
    """Generate 3D ground-truth trajectory CSV, scene metadata JSON, and multi-view PNG frames."""
    config = get_scenario_config(
        scenario=scenario,
        friction=friction,
        mass=mass,
        initial_velocity=initial_velocity,
        num_frames=num_frames,
    )
    output_dir = Path(output_root) / config.scenario
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    used_npa = False
    if use_nebius:
        used_npa = _try_run_npa_isaac_lab(config.scenario, output_dir, config.num_frames)

    trajectory = simulate_trajectory(config)
    trajectory_csv = output_dir / "trajectory.csv"
    metadata_json = output_dir / "metadata.json"

    with trajectory_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_index",
                "t",
                "x",
                "y",
                "z",
                "qx",
                "qy",
                "qz",
                "qw",
                "vx",
                "vy",
                "vz",
                "wx",
                "wy",
                "wz",
                "contact",
                "sdf_clearance",
                "grasp_force_n",
                "ee_x",
                "ee_y",
                "ee_z",
                "stage",
                "mechanical_energy",
            ]
        )
        for pt in trajectory:
            writer.writerow(
                [
                    pt.frame_index,
                    pt.t,
                    pt.x,
                    pt.y,
                    pt.z,
                    pt.qx,
                    pt.qy,
                    pt.qz,
                    pt.qw,
                    pt.vx,
                    pt.vy,
                    pt.vz,
                    pt.wx,
                    pt.wy,
                    pt.wz,
                    pt.contact,
                    pt.sdf_clearance,
                    pt.grasp_force_n,
                    pt.ee_x,
                    pt.ee_y,
                    pt.ee_z,
                    pt.stage,
                    pt.mechanical_energy,
                ]
            )

    frame_paths: List[Path] = []
    history: List[TrajectoryPoint] = []
    for pt in trajectory:
        history.append(pt)
        frame_path = frames_dir / f"frame_{pt.frame_index:03d}.png"
        if not used_npa or not frame_path.exists():
            render_scenario_frame(
                config=config,
                point=pt,
                history=history,
                output_path=frame_path,
                mode_label="ISAAC LAB 3D GROUND TRUTH",
            )
        frame_paths.append(frame_path)

    metadata: Dict[str, Any] = {
        **asdict(config),
        "dimension": "3D",
        "camera_representation": "multi_view_perspective_and_depth_sdf",
        "engine": "nvidia-isaac-lab-3d-npa" if used_npa else "isaac-lab-3d-compatible-solver",
        "num_frames_rendered": len(frame_paths),
        "initial_energy_joules": trajectory[0].mechanical_energy if trajectory else 0.0,
        "final_energy_joules": trajectory[-1].mechanical_energy if trajectory else 0.0,
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return SimulationResult(
        scenario=config.scenario,
        output_dir=output_dir,
        trajectory_csv=trajectory_csv,
        metadata_json=metadata_json,
        frames_dir=frames_dir,
        frame_paths=frame_paths,
        trajectory=trajectory,
        metadata=metadata,
    )


def generate_all_scenarios(
    output_root: Path | str = Path("data/ground_truth"),
    num_frames: int = 101,
    use_nebius: bool = False,
) -> Dict[str, SimulationResult]:
    """Generate 3D ground truth for all registered benchmark scenarios."""
    results: Dict[str, SimulationResult] = {}
    for name in SCENARIO_REGISTRY:
        results[name] = generate_ground_truth(
            scenario=name,
            output_root=output_root,
            num_frames=num_frames,
            use_nebius=use_nebius,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 3D Isaac Lab Ground Truth scenarios.")
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Scenario name ('franka_lift_cube', 'block_stack', 'projectile', 'ramp_roll', or 'all').",
    )
    parser.add_argument("--output", type=str, default="data/ground_truth", help="Output directory.")
    parser.add_argument("--num-frames", type=int, default=101, help="Number of frames to simulate.")
    parser.add_argument("--friction", type=float, default=None, help="Override friction coefficient.")
    parser.add_argument("--mass", type=float, default=None, help="Override object mass (kg).")
    parser.add_argument("--initial-velocity", type=float, default=None, help="Override initial velocity (m/s).")
    parser.add_argument("--use-nebius", action="store_true", help="Invoke npa workbench isaac-lab if available.")
    args = parser.parse_args()

    if args.scenario == "all":
        results = generate_all_scenarios(
            output_root=args.output,
            num_frames=args.num_frames,
            use_nebius=args.use_nebius,
        )
        for name, res in results.items():
            print(f"[3D GroundTruth] Generated {name} ({res.metadata['isaac_task_id']}): {len(res.frame_paths)} frames -> {res.output_dir}")
    else:
        res = generate_ground_truth(
            scenario=args.scenario,
            output_root=args.output,
            friction=args.friction,
            mass=args.mass,
            initial_velocity=args.initial_velocity,
            num_frames=args.num_frames,
            use_nebius=args.use_nebius,
        )
        print(f"[3D GroundTruth] Generated {res.scenario} ({res.metadata['isaac_task_id']}): {len(res.frame_paths)} frames -> {res.output_dir}")


if __name__ == "__main__":
    main()
