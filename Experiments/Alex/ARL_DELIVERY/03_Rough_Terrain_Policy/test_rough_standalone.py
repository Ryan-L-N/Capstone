"""
Standalone rough-terrain policy test with proper resets.

Tests:
  A) Manual PD torques hold default pose (no policy) — baseline
  B) Policy + manual PD torques from clean state — deployment method
  C) Policy + PhysX position drive (default USD gains) — comparison

Each test does a FULL robot reset (body pose + joints + velocities).
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from isaacsim.core.api import World
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from omni.isaac.quadruped.robots import SpotFlatTerrainPolicy

# Tee stdout to a log file so output is captured even when Isaac Sim swallows it
_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_rough_results.log")
class _Tee:
    def __init__(self, path, orig):
        self._file = open(path, "w")
        self._orig = orig
    def write(self, s):
        self._file.write(s)
        self._file.flush()
        self._orig.write(s)
    def flush(self):
        self._file.flush()
        self._orig.flush()
sys.stdout = _Tee(_LOG_PATH, sys.stdout)

PHYSICS_DT = 1.0 / 500.0
RENDERING_DT = 10.0 / 500.0

world = World(physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT, stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

START_POS = np.array([0.0, 0.0, 0.60])
spot_flat = SpotFlatTerrainPolicy(
    prim_path="/World/Spot", name="Spot", position=START_POS,
)

# Load rough policy
CKPT = r"C:\IsaacLab\logs\rsl_rl\spot_rough\2026-02-09_15-18-50\model_4999.pt"
actor = nn.Sequential(
    nn.Linear(235, 512), nn.ELU(),
    nn.Linear(512, 256), nn.ELU(),
    nn.Linear(256, 128), nn.ELU(),
    nn.Linear(128, 12),
)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
actor.load_state_dict({k.replace("actor.", ""): v
    for k, v in ckpt["model_state_dict"].items() if k.startswith("actor.")})
actor.eval()
print(f"[TEST] Actor loaded")

# Constants
TRAINING_DEFAULTS = {
    "fl_hx":  0.1, "fr_hx": -0.1, "hl_hx":  0.1, "hr_hx": -0.1,
    "fl_hy":  0.9, "fr_hy":  0.9, "hl_hy":  1.1, "hr_hy":  1.1,
    "fl_kn": -1.5, "fr_kn": -1.5, "hl_kn": -1.5, "hr_kn": -1.5,
}
HIP_EFFORT = 45.0
ACTION_SCALE = 0.25
DECIMATION = 10
KP, KD = 60.0, 1.5

KNEE_ANGLES = np.array([
    -2.7929, -2.7421, -2.6913, -2.6406, -2.5898, -2.5390, -2.4883,
    -2.4375, -2.3867, -2.3360, -2.2852, -2.2344, -2.1836, -2.1329,
    -2.0821, -2.0313, -1.9806, -1.9298, -1.8790, -1.8283, -1.7775,
    -1.7267, -1.6760, -1.6252, -1.5744, -1.5237, -1.4729, -1.4221,
    -1.3714, -1.3206, -1.2698, -1.2190, -1.1683, -1.1175, -1.0667,
    -1.0160, -0.9652, -0.9144, -0.8637, -0.8129, -0.7621, -0.7114,
    -0.6606, -0.6098, -0.5590, -0.5083, -0.4575, -0.4067, -0.3560,
    -0.3052, -0.2544, -0.2471,
])
KNEE_VALUES = np.array([
    37.17, 39.44, 41.83, 43.87, 46.03, 48.02, 49.97,
    51.79, 53.45, 56.31, 58.89, 61.20, 63.28, 66.68,
    69.92, 72.89, 75.67, 78.19, 80.55, 83.37, 86.07,
    88.76, 91.62, 94.35, 97.13, 100.47, 103.43, 106.42,
    108.96, 111.16, 112.98, 113.12, 113.24, 112.47, 111.70,
    110.68, 109.56, 108.00, 107.14, 105.80, 103.09, 100.36,
    96.27, 91.07, 84.87, 78.24, 69.59, 60.42, 51.42,
    41.65, 31.60, 30.60,
])

# Initialize
world.reset()
spot_flat.initialize()
robot = spot_flat.robot
av = robot._articulation_view
n_dof = robot.num_dof

# Build training default positions
dof_names = robot.dof_names
default_pos = robot.get_joint_positions().copy()
hip_idx, knee_idx = [], []
for i, name in enumerate(dof_names):
    if name in TRAINING_DEFAULTS:
        default_pos[i] = TRAINING_DEFAULTS[name]
    if name.endswith('_hx') or name.endswith('_hy'):
        hip_idx.append(i)
    elif name.endswith('_kn'):
        knee_idx.append(i)

# Save default USD gains (from flat policy)
saved_kps, saved_kds = av.get_gains()
print(f"[TEST] DOF order: {list(dof_names)}")
print(f"[TEST] Default pos: {np.array2string(default_pos, precision=3)}")
print(f"[TEST] USD default Kp range: [{saved_kps.min():.1f}, {saved_kps.max():.1f}]")
print(f"[TEST] USD default Kd range: [{saved_kds.min():.1f}, {saved_kds.max():.1f}]")
print(f"[TEST] Hip DOFs: {hip_idx}  Knee DOFs: {knee_idx}")

# ===========================================================================
# CRITICAL FIX: Match articulation solver iterations from SPOT_CFG
# Training uses solver_position_iteration_count=4, velocity=0
# USD default is typically 32/32 which produces DIFFERENT dynamics
# ===========================================================================
try:
    old_pos_iters = av.get_solver_position_iteration_counts()
    old_vel_iters = av.get_solver_velocity_iteration_counts()
    print(f"\n[TEST] Solver iters BEFORE: pos={old_pos_iters}, vel={old_vel_iters}")

    av.set_solver_position_iteration_counts(np.array([4]))
    av.set_solver_velocity_iteration_counts(np.array([0]))

    new_pos_iters = av.get_solver_position_iteration_counts()
    new_vel_iters = av.get_solver_velocity_iteration_counts()
    print(f"[TEST] Solver iters AFTER:  pos={new_pos_iters}, vel={new_vel_iters}")
except Exception as e:
    print(f"[TEST] WARNING: Could not set solver iterations: {e}")

# Enable self-collisions (SPOT_CFG: enabled_self_collisions=True)
try:
    av.set_enabled_self_collisions(np.array([True]))
    print(f"[TEST] Self-collisions enabled")
except Exception as e:
    print(f"[TEST] Could not set self-collisions: {e}")

# Set rigid body properties to match SPOT_CFG
try:
    av.set_linear_damping(np.array([0.0]))
    av.set_angular_damping(np.array([0.0]))
    av.set_max_depenetration_velocity(np.array([1.0]))
    print(f"[TEST] Rigid body props: damping=0, max_depenetration=1.0")
except Exception as e:
    print(f"[TEST] Could not set rigid body props: {e}")

print()


def full_reset():
    """Reset robot body + joints + velocities to upright default pose."""
    robot.set_world_pose(position=START_POS, orientation=np.array([1, 0, 0, 0]))
    robot.set_linear_velocity(np.zeros(3))
    robot.set_angular_velocity(np.zeros(3))
    robot.set_joint_positions(default_pos)
    robot.set_joint_velocities(np.zeros(n_dof))
    # Step once to apply reset
    world.step(render=False)


def clamp_torques(torques, joint_pos):
    """Apply per-joint effort limits."""
    t = torques.copy()
    for i in hip_idx:
        t[i] = np.clip(t[i], -HIP_EFFORT, HIP_EFFORT)
    for i in knee_idx:
        tau_max = np.interp(float(joint_pos[i]), KNEE_ANGLES, KNEE_VALUES)
        t[i] = np.clip(t[i], -tau_max, tau_max)
    return t


def run_policy(obs_state, action_state, command):
    """Evaluate policy and return action. Modifies obs_state in-place."""
    lin_vel_I = robot.get_linear_velocity()
    ang_vel_I = robot.get_angular_velocity()
    pos, q_IB = robot.get_world_pose()
    R_BI = quat_to_rot_matrix(q_IB).T
    lin_vel_b = R_BI @ lin_vel_I
    ang_vel_b = R_BI @ ang_vel_I
    gravity_b = R_BI @ np.array([0.0, 0.0, -1.0])

    obs = np.zeros(235, dtype=np.float32)
    obs[0:3] = lin_vel_b
    obs[3:6] = ang_vel_b
    obs[6:9] = gravity_b
    obs[9:12] = command
    jp = robot.get_joint_positions()
    jv = robot.get_joint_velocities()
    obs[12:24] = jp - default_pos
    obs[24:36] = jv
    obs[36:48] = action_state['prev']
    obs[48:235] = 1.0  # height scan

    with torch.no_grad():
        t = torch.from_numpy(obs).unsqueeze(0).float()
        action = actor(t).squeeze(0).numpy()

    action_state['prev'] = action.copy()
    return action, float(pos[2]), gravity_b


def print_status(label, step, eval_idx, action, body_z, gravity_b):
    rolled = " ROLLED!" if body_z < 0.2 else ""
    if eval_idx < 15 or eval_idx % 25 == 0:
        print(f"  [{label}] step={step:5d} eval={eval_idx:3d} | "
              f"act_norm={np.linalg.norm(action):.2f} "
              f"range=[{action.min():+.3f},{action.max():+.3f}] | "
              f"z={body_z:.3f} | "
              f"g=[{gravity_b[0]:+.3f},{gravity_b[1]:+.3f},{gravity_b[2]:+.3f}]"
              f"{rolled}")


# =====================================================================
# TEST A: Manual PD torques hold default pose (NO policy)
# =====================================================================
print("\n" + "="*60)
print("  TEST A: Manual PD hold default pose (no policy)")
print("  PhysX Kp=0, Kd=0, Python PD Kp=60, Kd=1.5")
print("="*60)

full_reset()
# Set PhysX to torque mode
av.set_gains(kps=np.zeros((1, n_dof)), kds=np.zeros((1, n_dof)))
try: av.set_friction_coefficients(np.zeros((1, n_dof)))
except: pass
try: av.set_armatures(np.zeros((1, n_dof)))
except: pass
try: av.set_max_efforts(np.full((1, n_dof), 1e9))
except: pass
try: av.set_max_joint_velocities(np.full((1, n_dof), 12.0))
except: pass

a_fell = False
for step in range(2500):  # 5 seconds
    cur_pos = robot.get_joint_positions()
    cur_vel = robot.get_joint_velocities()
    torques = KP * (default_pos - cur_pos) - KD * cur_vel
    torques = clamp_torques(torques, cur_pos)
    robot.set_joint_efforts(torques)
    world.step(render=False)

    if step % 250 == 0 or step == 2499:
        pos, q = robot.get_world_pose()
        bz = float(pos[2])
        g = quat_to_rot_matrix(q).T @ np.array([0.0, 0.0, -1.0])
        print(f"  [A] step={step:5d} | z={bz:.3f} | "
              f"g=[{g[0]:+.3f},{g[1]:+.3f},{g[2]:+.3f}]")
        if bz < 0.1:
            print(f"\n[TEST-A] FELL at step {step}!")
            a_fell = True
            break

if not a_fell:
    print(f"[TEST-A] SURVIVED 5s!")


# =====================================================================
# TEST B: Policy + Manual PD torques (deployment method)
# =====================================================================
print("\n" + "="*60)
print("  TEST B: Policy + Manual PD torques")
print("  PhysX Kp=0, Kd=0, Python PD Kp=60, Kd=1.5")
print("="*60)

full_reset()
# Re-apply torque mode settings
av.set_gains(kps=np.zeros((1, n_dof)), kds=np.zeros((1, n_dof)))
try: av.set_friction_coefficients(np.zeros((1, n_dof)))
except: pass
try: av.set_armatures(np.zeros((1, n_dof)))
except: pass
try: av.set_max_efforts(np.full((1, n_dof), 1e9))
except: pass
try: av.set_max_joint_velocities(np.full((1, n_dof), 12.0))
except: pass

# Settle with manual PD for 200 steps
for _ in range(200):
    cur_pos = robot.get_joint_positions()
    cur_vel = robot.get_joint_velocities()
    torques = KP * (default_pos - cur_pos) - KD * cur_vel
    torques = clamp_torques(torques, cur_pos)
    robot.set_joint_efforts(torques)
    world.step(render=False)

pos, _ = robot.get_world_pose()
print(f"[TEST-B] After settle: z={float(pos[2]):.3f}")

action = np.zeros(12)
act_state = {'prev': np.zeros(12)}
command = [0.0, 0.0, 0.0]
b_fell = False

for step in range(2500):  # 5 seconds
    if step % DECIMATION == 0:
        eval_idx = step // DECIMATION
        action, body_z, gravity_b = run_policy(None, act_state, command)
        print_status("B", step, eval_idx, action, body_z, gravity_b)
        if body_z < 0.1:
            print(f"\n[TEST-B] FELL at step {step}!")
            b_fell = True
            break

    target = default_pos + action * ACTION_SCALE
    cur_pos = robot.get_joint_positions()
    cur_vel = robot.get_joint_velocities()
    torques = KP * (target - cur_pos) - KD * cur_vel
    torques = clamp_torques(torques, cur_pos)
    robot.set_joint_efforts(torques)
    world.step(render=True)

if not b_fell:
    pos, _ = robot.get_world_pose()
    print(f"\n[TEST-B] SURVIVED 5s! z={float(pos[2]):.3f}")


# =====================================================================
# TEST C: Policy + PhysX position drive (default USD gains)
# =====================================================================
print("\n" + "="*60)
print("  TEST C: Policy + PhysX POSITION DRIVE (USD default gains)")
print("="*60)

full_reset()
# Restore default USD gains
av.set_gains(kps=saved_kps, kds=saved_kds)
# Restore normal effort limits
try: av.set_max_efforts(np.full((1, n_dof), 1000.0))
except: pass
try: av.set_max_joint_velocities(np.full((1, n_dof), 100.0))
except: pass

# Settle with position drive
for _ in range(200):
    robot.set_joint_positions(default_pos)
    world.step(render=False)

pos, _ = robot.get_world_pose()
print(f"[TEST-C] After settle: z={float(pos[2]):.3f}")

action = np.zeros(12)
act_state = {'prev': np.zeros(12)}
c_fell = False

for step in range(2500):  # 5 seconds
    if step % DECIMATION == 0:
        eval_idx = step // DECIMATION
        action, body_z, gravity_b = run_policy(None, act_state, command)
        print_status("C", step, eval_idx, action, body_z, gravity_b)
        if body_z < 0.1:
            print(f"\n[TEST-C] FELL at step {step}!")
            c_fell = True
            break

    target = default_pos + action * ACTION_SCALE
    robot.set_joint_positions(target)
    world.step(render=True)

if not c_fell:
    pos, _ = robot.get_world_pose()
    print(f"\n[TEST-C] SURVIVED 5s! z={float(pos[2]):.3f}")


# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "="*60)
print("  RESULTS SUMMARY")
print("="*60)
print(f"  A (Manual PD hold, no policy): {'FELL' if a_fell else 'SURVIVED'}")
print(f"  B (Manual PD + policy):        {'FELL' if b_fell else 'SURVIVED'}")
print(f"  C (PhysX pos drive + policy):  {'FELL' if c_fell else 'SURVIVED'}")
print()
print("  If A fails: Manual PD torque mode itself is broken")
print("  If A ok, B fails: Policy outputs are problematic")
print("  If B fails, C ok: Torque mode is the issue")
print("  If both B,C fail: Policy is the issue")
print("="*60)

for _ in range(300):
    world.step(render=True)

simulation_app.close()
