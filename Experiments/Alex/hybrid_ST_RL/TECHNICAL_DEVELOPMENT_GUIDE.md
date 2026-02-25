# Technical Development Guide: Spot Quadruped RL Training Pipeline

**Isaac Sim 5.1.0.0 / Isaac Lab 0.54.2 + RSL-RL — Code Patterns and API Reference**

*AI2C Tech Capstone — February 2026*

---

This document is the **code-level developer guide** for the hybrid ST-RL training pipeline. It covers Isaac Sim API patterns, reproducible code snippets, configuration walkthrough, and production ops. For the academic methodology, see [TRAINING_METHODOLOGY.md](TRAINING_METHODOLOGY.md). For a plain-language explanation, see [HOW_WE_TRAINED_SPOT.md](HOW_WE_TRAINED_SPOT.md).

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Project Layout](#2-project-layout)
3. [The AppLauncher Pattern](#3-the-applauncher-pattern)
4. [The @configclass System](#4-the-configclass-system)
5. [Gymnasium Registration + RSL-RL Integration](#5-gymnasium-registration--rsl-rl-integration)
6. [Terrain System](#6-terrain-system)
7. [Reward Engineering](#7-reward-engineering)
8. [Progressive Domain Randomization](#8-progressive-domain-randomization)
9. [Checkpoint Loading Fixes](#9-checkpoint-loading-fixes)
10. [The Monkey-Patch Pattern](#10-the-monkey-patch-pattern)
11. [Teacher-Student Distillation](#11-teacher-student-distillation)
12. [Launch Scripts and Production Ops](#12-launch-scripts-and-production-ops)
13. [Deployment and Evaluation](#13-deployment-and-evaluation)
14. [Gotchas and Pitfalls](#14-gotchas-and-pitfalls)
15. [PPO Hyperparameters](#15-ppo-hyperparameters)
16. [Appendices](#16-appendices)

---

## 1. Quick Start

### Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| Isaac Sim | 5.1.0.0 | NVIDIA Omniverse |
| Isaac Lab | 0.54.2 | `isaaclab.sh` / `isaaclab.bat` |
| RSL-RL | latest | PPO implementation |
| PyTorch | 2.7.0+cu128 | TF32 enabled for H100 |
| Python | 3.11.x | Via conda env |
| GPU (dev) | RTX 2000 Ada (8 GB) | Local smoke tests |
| GPU (prod) | H100 NVL (96 GB) | Production training |

### Local Smoke Test (RTX 2000 Ada)

```bash
cd C:\IsaacLab
set OMNI_KIT_ACCEPT_EULA=YES
set PYTHONUNBUFFERED=1

isaaclab.bat -p /path/to/train_finetune.py --headless ^
    --num_envs 64 ^
    --max_iterations 10 ^
    --dr_expansion_iters 5 ^
    --actor_freeze_iters 3 ^
    --lr_warmup_iters 2 ^
    --min_noise_std 0.4 ^
    --max_noise_std 1.5 ^
    --seed 42 ^
    --checkpoint /path/to/model_27500.pt
```

**Expected output markers** (verify these appear):
- `SKIPPED N critic keys (intentionally reset)` — actor-only load working
- `Actor FROZEN for 3 iterations` — critic warmup active
- `Actor MLP UNFROZEN (noise std stays FROZEN at 0.65)` — warmup → LR warmup transition
- `LR warmup: 2.0e-06 → 1.0e-05 over N iterations` — gradual LR ramp active
- `dr=X.X%` in progress lines — progressive DR active
- `noise=0.650` — noise locked at 0.65 throughout (permanently frozen)
- `lr=X.Xe-0X` in progress lines — LR warmup visible
- No `NaN` in any reward terms

### H100 Production Launch

```bash
ssh t2user@ai2ct2
bash ~/hybrid_ST_RL/scripts/train_finetune_h100.sh
```

Or with overrides:

```bash
NUM_ENVS=8192 MAX_ITERS=30000 bash ~/hybrid_ST_RL/scripts/train_finetune_h100.sh
```

---

## 2. Project Layout

```
hybrid_ST_RL/
├── train_finetune.py           # Stage 1: Progressive fine-tuning (main training script)
├── train_from_scratch.py       # Attempt #5: from-scratch training (no checkpoint)
├── train_teacher.py            # Stage 2a: Teacher training with privileged obs
├── train_distill.py            # Stage 2b: Student distillation from teacher
│
├── configs/
│   ├── __init__.py
│   ├── finetune_env_cfg.py     # Stage 1 environment config (235-dim obs, 19 rewards, 12 terrains)
│   ├── finetune_ppo_cfg.py     # Stage 1 PPO hyperparameters ([512,256,128], LR=1e-5, Attempt #4)
│   ├── scratch_terrain_cfg.py  # Attempt #5 terrain curriculum (7 types, flat start)
│   ├── scratch_ppo_cfg.py      # Attempt #5 from-scratch PPO (LR=1e-3, standard)
│   ├── scratch_env_cfg.py      # Attempt #5 env config (235-dim, terrain curriculum)
│   ├── teacher_env_cfg.py      # Stage 2a environment config (254-dim obs = 235 + 19 privileged)
│   ├── teacher_ppo_cfg.py      # Stage 2a PPO hyperparameters
│   └── terrain_cfg.py          # ROBUST_TERRAINS_CFG: 12 terrain types, 10x40 grid
│
├── rewards/
│   ├── __init__.py
│   └── reward_terms.py         # 5 custom reward functions (vegetation drag, velocity modulation, etc.)
│
├── scripts/
│   ├── train_finetune_h100.sh  # H100 production launch (screen + tee + tensorboard)
│   ├── train_teacher_h100.sh   # H100 teacher launch
│   ├── train_distill_h100.sh   # H100 distillation launch
│   └── train_local_debug.sh    # Local smoke test (64 envs, 10 iters)
│
├── checkpoints/                # Place model_27500.pt here for local testing
├── logs/                       # Training logs (auto-created)
│
├── TRAINING_METHODOLOGY.md     # Academic methodology document
├── HOW_WE_TRAINED_SPOT.md      # Plain-language explanation
├── TECHNICAL_DEVELOPMENT_GUIDE.md  # This document
└── README.md
```

### Data Flow

```
model_27500.pt (48hr rough policy)
        │
        ▼
┌─────────────────────────┐     finetune_env_cfg.py ──┐
│  train_finetune.py      │◄────finetune_ppo_cfg.py ──┤
│  (Stage 1)              │◄────terrain_cfg.py ────────┤
│  235-dim obs, 19 rewards│◄────reward_terms.py ───────┘
└────────┬────────────────┘
         │ stage1_best.pt
         ▼
┌─────────────────────────┐     teacher_env_cfg.py ────┐
│  train_teacher.py       │◄────teacher_ppo_cfg.py ────┘
│  (Stage 2a)             │     (weight surgery: 235→254)
│  254-dim obs (privileged)│
└────────┬────────────────┘
         │ stage2a_best.pt
         ▼
┌─────────────────────────┐
│  train_distill.py       │◄─── stage1_best.pt (student)
│  (Stage 2b)             │◄─── stage2a_best.pt (teacher)
│  BC loss + PPO, 235-dim │
└────────┬────────────────┘
         │ final_student.pt
         ▼
   Deployment (4_env_test)
```

---

## 3. The AppLauncher Pattern

Isaac Sim has a strict initialization order: `SimulationApp` **must** exist before importing any `omni.*`, `pxr.*`, or Isaac Lab modules that depend on the Omniverse runtime. Violating this order causes cryptic segfaults.

### The Canonical Pattern

From `train_finetune.py` lines 32–89:

```python
# ── 0. Parse args BEFORE any Isaac imports ──────────────────────────────
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stage 1: Progressive fine-tuning")
parser.add_argument("--num_envs", type=int, default=16384)
parser.add_argument("--max_iterations", type=int, default=25000)
parser.add_argument("--checkpoint", type=str, default=None)
# ... more args ...
AppLauncher.add_app_launcher_args(parser)   # Adds --headless, --device, etc.
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]                    # CRITICAL: reset argv so gym doesn't choke

app_launcher = AppLauncher(args_cli)        # Creates SimulationApp
simulation_app = app_launcher.app

# ── 1. Imports (AFTER SimulationApp) ────────────────────────────────────
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401  — triggers standard gym registrations

# Enable TF32 for faster matmul on H100
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### Key Details

| Step | Why It Matters |
|------|---------------|
| `AppLauncher.add_app_launcher_args(parser)` | Injects `--headless`, `--device`, `--experience` flags |
| `args_cli, _ = parser.parse_known_args()` | `parse_known_args` tolerates unknown flags from Isaac internals |
| `sys.argv = [sys.argv[0]]` | Gymnasium's env checker crashes on unknown CLI args if you skip this |
| `app_launcher = AppLauncher(args_cli)` | Creates `SimulationApp` — after this line, omni imports are safe |
| `import isaaclab_tasks` | Registers Isaac Lab's built-in environments with Gymnasium |
| `torch.backends.cuda.matmul.allow_tf32 = True` | ~2x matmul throughput on H100 (Ampere+), negligible precision loss |

### Adding Custom Modules to Path

Every training script adds the project directory to `sys.path` so Python finds our custom configs and rewards:

```python
import os, sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from configs.finetune_env_cfg import SpotFinetuneEnvCfg
from configs.finetune_ppo_cfg import SpotFinetunePPORunnerCfg
```

---

## 4. The @configclass System

Isaac Lab uses a `@configclass` decorator (built on `dataclasses`) with `__post_init__` chaining for hierarchical configuration. Every aspect of the environment — observations, actions, rewards, events, terrains — is defined as a configclass.

### Environment Config Architecture

`SpotFinetuneEnvCfg` (from `configs/finetune_env_cfg.py`) contains 7 manager sub-configs:

```python
@configclass
class SpotFinetuneEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: SpotFinetuneObservationsCfg = SpotFinetuneObservationsCfg()
    actions:      SpotFinetuneActionsCfg      = SpotFinetuneActionsCfg()
    commands:     SpotFinetuneCommandsCfg      = SpotFinetuneCommandsCfg()
    rewards:      SpotFinetuneRewardsCfg       = SpotFinetuneRewardsCfg()
    terminations: SpotFinetuneTerminationsCfg  = SpotFinetuneTerminationsCfg()
    events:       SpotFinetuneEventCfg         = SpotFinetuneEventCfg()
    curriculum:   SpotFinetuneCurriculumCfg    = SpotFinetuneCurriculumCfg()
```

### Observations — 235 Dimensions

The observation space is 48 proprioceptive + 187 height scan = 235 total:

```python
@configclass
class SpotFinetuneObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Proprioceptive (48 dims):
        base_lin_vel    = ObsTerm(func=mdp.base_lin_vel,    noise=Unoise(n_min=-0.15, n_max=0.15))  # 3
        base_ang_vel    = ObsTerm(func=mdp.base_ang_vel,    noise=Unoise(n_min=-0.15, n_max=0.15))  # 3
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))  # 3
        velocity_commands = ObsTerm(func=mdp.generated_commands)                                     # 3
        joint_pos       = ObsTerm(func=mdp.joint_pos_rel,   noise=Unoise(n_min=-0.05, n_max=0.05))  # 12
        joint_vel       = ObsTerm(func=mdp.joint_vel_rel,   noise=Unoise(n_min=-0.5,  n_max=0.5))   # 12
        actions         = ObsTerm(func=mdp.last_action)                                              # 12

        # Exteroceptive (187 dims):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.15, n_max=0.15),
            clip=(-1.0, 1.0),
        )  # 17x11 grid = 187

        def __post_init__(self):
            self.enable_corruption = True   # Apply noise during training
            self.concatenate_terms = True   # Stack into single tensor
```

### Actions — 12 Dimensions

```python
@configclass
class SpotFinetuneActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],        # All 12 joints
        scale=0.25,                # Actions scaled by 0.25 rad
        use_default_offset=True,   # Centered around default joint positions
    )
```

### Physics Configuration

The `__post_init__` method configures simulation physics:

```python
def __post_init__(self):
    super().__post_init__()

    # Physics — 500 Hz with decimation=10 → 50 Hz control
    self.decimation = 10
    self.episode_length_s = 30.0      # 50% longer than 48hr's 20s
    self.sim.dt = 0.002               # 500 Hz physics
    self.sim.render_interval = self.decimation

    # GPU PhysX buffers — sized for 8K+ envs with 12 terrain types
    self.sim.physx.gpu_collision_stack_size = 2**30     # 1 GB
    self.sim.physx.gpu_max_rigid_contact_count = 2**23
    self.sim.physx.gpu_max_rigid_patch_count = 2**23

    # Height scanner — 17x11 grid = 187 dims, 0.1m resolution
    self.scene.height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        mesh_prim_paths=["/World/ground"],
    )

    # ROBUST terrain — 12 types, 400 patches, 10x40 grid
    self.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROBUST_TERRAINS_CFG,
        max_init_terrain_level=5,
    )
```

---

## 5. Gymnasium Registration + RSL-RL Integration

Isaac Lab environments are registered with Gymnasium at runtime, then wrapped for RSL-RL compatibility.

### The Registration Pattern

From `train_finetune.py` lines 306–320:

```python
# Register our custom environment with Gymnasium
gym.register(
    id="Isaac-Velocity-Finetune-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,    # Isaac Lab envs don't pass Gymnasium's built-in checks
    kwargs={
        "env_cfg_entry_point": f"{SpotFinetuneEnvCfg.__module__}:{SpotFinetuneEnvCfg.__name__}",
    },
)

# Create the environment
env = gym.make("Isaac-Velocity-Finetune-Spot-v0", cfg=env_cfg)

# Wrap for RSL-RL (converts obs/actions to RSL-RL's expected format)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

# Create the RSL-RL training runner
runner = OnPolicyRunner(
    env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
)
```

### Why `disable_env_checker=True`

Gymnasium's default environment checker validates observation/action spaces against strict typing rules. Isaac Lab's `ManagerBasedRLEnv` uses custom tensor-based spaces that fail these checks. The environment works correctly — the checker is simply incompatible.

### The RSL-RL Pipeline

```
gym.register() → gym.make() → RslRlVecEnvWrapper → OnPolicyRunner
                                    │                      │
                            Converts obs/act tensors   Runs PPO:
                            to RSL-RL format           - Rollout collection
                                                       - Advantage estimation
                                                       - Policy update
                                                       - Logging + checkpointing
```

The `OnPolicyRunner` handles the full training loop. We call `runner.learn()` and it runs for `max_iterations` PPO updates, automatically saving checkpoints at `save_interval` steps.

---

## 6. Terrain System

### ROBUST_TERRAINS_CFG — 12 Terrain Types

From `configs/terrain_cfg.py`: a 10×40 grid (400 patches, each 8m×8m) with curriculum-based difficulty progression.

```python
ROBUST_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,       # 10 difficulty levels (curriculum progression)
    num_cols=40,       # 40 columns (terrain variety)
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,   # Robots promoted/demoted based on velocity tracking
    sub_terrains={...},
)
```

### Terrain Breakdown

| Category | Type | Proportion | Difficulty Range | Purpose |
|----------|------|-----------|-----------------|---------|
| **A: Geometric (40%)** | `pyramid_stairs_up` | 10% | 0.05–0.25m step | Ascending stair traversal |
| | `pyramid_stairs_down` | 10% | 0.05–0.25m step | Descending (inverted) |
| | `boxes` | 10% | 0.05–0.25m height | Rubble / boulder proxy |
| | `stepping_stones` | 5% | 0.25–0.5m width | Precise foot placement |
| | `gaps` | 5% | 0.1–0.5m width | Stride / jump training |
| **B: Surface (35%)** | `random_rough` | 10% | 0.02–0.15m noise | General uneven ground |
| | `hf_pyramid_slope_up` | 7.5% | 0.0–0.5 slope | Uphill traversal |
| | `hf_pyramid_slope_down` | 7.5% | 0.0–0.5 slope | Downhill traversal |
| | `wave_terrain` | 5% | 0.05–0.2m amplitude | Undulating ground |
| | `friction_plane` | 5% | Flat, low friction | Slippery surface training |
| | `vegetation_plane` | 5% | Flat + drag force | Grass/mud resistance |
| **C: Compound (25%)** | `hf_stairs_up` | 10% | 0.05–0.20m step | Noisy heightfield stairs |
| | `discrete_obstacles` | 5% | 0.05–0.30m height | Scattered blocks |
| | `repeated_boxes` | 5% | 0.05–0.20m height | Regular obstacle pattern |

### Special Planes

Two terrain types are flat planes with a specific training purpose:

- **`friction_plane`** (5%): Perfectly flat. The only challenge is low friction (mu down to 0.05). `VegetationDragReward` applies zero drag here. Teaches balance on slippery surfaces.

- **`vegetation_plane`** (5%): Perfectly flat. `VegetationDragReward` always applies drag > 0 (0.5–20.0 N·s/m). Teaches the policy to push through resistance.

### Terrain Curriculum

```python
@configclass
class SpotFinetuneCurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
```

The `terrain_levels_vel` function promotes robots to harder difficulty rows when they track commanded velocity well, and demotes them when they fail. This ensures the policy is always training at an appropriate challenge level.

---

## 7. Reward Engineering

### All 19 Reward Terms

From `configs/finetune_env_cfg.py`:

| Term | Weight | Type | Source |
|------|--------|------|--------|
| `base_linear_velocity` | +7.0 | Task | `spot_mdp` |
| `gait` | +10.0 | Task | `spot_mdp.GaitReward` |
| `base_angular_velocity` | +5.0 | Task | `spot_mdp` |
| `foot_clearance` | +3.5 | Task | `spot_mdp` |
| `air_time` | +3.0 | Task | `spot_mdp` |
| `velocity_modulation` | +2.0 | Task | **Custom** |
| `vegetation_drag` | -0.001 | Physics+Penalty | **Custom (class-based)** |
| `base_orientation` | -5.0 | Penalty | `spot_mdp` |
| `base_motion` | -4.0 | Penalty | `spot_mdp` |
| `foot_slip` | -3.0 | Penalty | `spot_mdp` |
| `action_smoothness` | -2.0 | Penalty | `spot_mdp` |
| `body_height_tracking` | -2.0 | Penalty | **Custom** |
| `stumble` | -2.0 | Penalty | **Custom** |
| `air_time_variance` | -1.0 | Penalty | `spot_mdp` |
| `joint_pos` | -1.0 | Penalty | `spot_mdp` |
| `contact_force_smoothness` | -0.5 | Penalty | **Custom** |
| `joint_vel` | -0.05 | Penalty | `spot_mdp` |
| `joint_torques` | -0.002 | Penalty | `spot_mdp` |
| `joint_acc` | -0.0005 | Penalty | `spot_mdp` |

### Custom Reward Deep Dives

#### VegetationDragReward (Class-Based Physics Modifier + Reward)

This is the most complex reward term — it's both a **physics modifier** (applies forces to the simulation) and a **reward function** (returns a penalty). It's implemented as a class inheriting from `ManagerTermBase`.

From `rewards/reward_terms.py`:

```python
class VegetationDragReward(ManagerTermBase):
    """Applies F_drag = -drag_coeff * v_foot to each foot every control step."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]

        # Build terrain-aware column masks (which columns are vegetation/friction)
        terrain_gen_cfg = getattr(env.scene.terrain.cfg, "terrain_generator", None)
        if terrain_gen_cfg is not None and terrain_gen_cfg.curriculum:
            # Map columns → sub-terrain indices
            sub_terrain_names = list(terrain_gen_cfg.sub_terrains.keys())
            proportions = np.array([sc.proportion for sc in terrain_gen_cfg.sub_terrains.values()])
            proportions = proportions / proportions.sum()
            cum_props = np.cumsum(proportions)

            col_to_idx = []
            for col in range(terrain_gen_cfg.num_cols):
                idx = int(np.min(np.where(cum_props > col / terrain_gen_cfg.num_cols)[0]))
                col_to_idx.append(idx)

            # Boolean masks: is column X a vegetation/friction plane?
            veg_idx = sub_terrain_names.index("vegetation_plane")
            fric_idx = sub_terrain_names.index("friction_plane")
            self.is_vegetation_col = torch.tensor([col_to_idx[c] == veg_idx for c in range(num_cols)], ...)
            self.is_friction_col = torch.tensor([col_to_idx[c] == fric_idx for c in range(num_cols)], ...)

        # Per-env drag coefficient: sampled at reset
        self.drag_coeff = torch.zeros(env.num_envs, 1, device=env.device)
```

**Tiered drag sampling** (at each episode reset):

```python
def _resample_drag(self, env_ids):
    # 25% clean (c=0), 25% light [0.5,5.0], 25% medium [5.0,12.0], 25% heavy [12.0,20.0]
    tier_rand = torch.rand(n, device=dev)
    # ... tier assignment ...

    # Terrain overrides:
    if self.terrain_aware:
        robot_cols = terrain.terrain_types[env_ids]
        drag_vals[self.is_friction_col[robot_cols]] = 0.0        # Friction plane: no drag
        drag_vals[self.is_vegetation_col[robot_cols]] = uniform(0.5, drag_max)  # Veg: always drag
```

**Physics application** (every control step at 50 Hz):

```python
def __call__(self, env, ...):
    foot_vel = self.asset.data.body_lin_vel_w[:, self.foot_body_ids, :]
    drag_force = -self.drag_coeff.unsqueeze(2) * foot_vel  # F = -c * v
    drag_force = drag_force * is_contact.unsqueeze(2).float()  # Only when touching ground
    drag_force[:, :, 2] = 0.0  # Horizontal only

    # Apply persistent forces via wrench composer (acts every physics sub-step)
    all_forces = torch.zeros(num_envs, self.asset.num_bodies, 3, device=env.device)
    all_forces[:, self.foot_body_ids, :] = drag_force
    self.asset.permanent_wrench_composer.set_forces_and_torques(forces=all_forces, torques=all_torques)

    return torch.sum(torch.norm(drag_force, dim=-1), dim=1)  # Penalty
```

#### velocity_modulation_reward — Adaptive Speed Target

Prevents the policy from freezing on hard terrain or charging recklessly on easy terrain:

```python
def velocity_modulation_reward(env, asset_cfg, sensor_cfg, std=0.5):
    cmd_speed = torch.linalg.norm(cmd_vel, dim=1)
    actual_speed = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)

    # Estimate difficulty from contact force variance
    force_variance = torch.var(force_magnitudes, dim=1)
    difficulty_factor = torch.clamp(force_variance / 500.0, 0.0, 1.0)

    # Adaptive target: 100% on easy terrain, 50% on very hard terrain
    adaptive_target = cmd_speed * (1.0 - 0.5 * difficulty_factor)

    speed_error = torch.abs(actual_speed - adaptive_target)
    return torch.exp(-speed_error / std)
```

#### body_height_tracking_penalty — Height Regulation

Prevents unnatural crouching or rising:

```python
def body_height_tracking_penalty(env, asset_cfg, target_height=0.42):
    body_height = asset.data.root_pos_w[:, 2]
    return torch.square(body_height - target_height)  # L2 penalty
```

#### contact_force_smoothness_penalty — Gentle Foot Placement

Penalizes sudden GRF spikes (slamming feet down):

```python
def contact_force_smoothness_penalty(env, sensor_cfg):
    current_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    prev_forces = contact_sensor.data.net_forces_w_history[:, 1, sensor_cfg.body_ids]
    force_diff = torch.norm(current_forces - prev_forces, dim=-1)
    return torch.sum(force_diff, dim=1)
```

#### stumble_penalty — Tripping Detection

Fires when a foot is elevated AND has significant contact force (hitting an obstacle's side):

```python
def stumble_penalty(env, asset_cfg, sensor_cfg, knee_height=0.15, force_threshold=5.0):
    foot_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    force_mags = torch.norm(net_forces, dim=-1)
    is_stumble = (foot_heights > knee_height) & (force_mags > force_threshold)
    return torch.sum(is_stumble.float() * force_mags, dim=1)
```

---

## 8. Progressive Domain Randomization

Instead of training with full-difficulty DR from the start (which can destroy the warm-started policy), we linearly expand DR ranges over 15,000 iterations.

### DR Schedule Definition

From `train_finetune.py` lines 100–118:

```python
DR_SCHEDULE = {
    # Friction                       (start, end)
    "static_friction_min":  (0.3,  0.1),    # Easy → ice-like
    "static_friction_max":  (1.3,  1.5),    # Moderate → rubber
    "dynamic_friction_min": (0.25, 0.08),
    "dynamic_friction_max": (1.1,  1.2),
    # Push robot
    "push_velocity":        (0.5,  1.0),    # Gentle → aggressive pushes
    "push_interval_min":    (10.0, 6.0),    # Infrequent → frequent
    "push_interval_max":    (15.0, 13.0),
    # External forces
    "ext_force":            (3.0,  6.0),    # Mild → strong winds
    "ext_torque":           (1.0,  2.5),
    # Mass
    "mass_offset":          (5.0,  7.0),    # Moderate → heavy payload
    # Joint velocity reset
    "joint_vel_range":      (2.5,  3.0),
}
```

### Linear Interpolation

```python
def lerp(start: float, end: float, fraction: float) -> float:
    fraction = max(0.0, min(1.0, fraction))
    return start + (end - start) * fraction
```

### Applying DR Each Iteration

The `update_dr_params()` function modifies the environment config in-place. Because events use `mode="reset"`, the new ranges take effect when environments reset:

```python
def update_dr_params(env, iteration, expansion_iters):
    fraction = min(iteration / max(expansion_iters, 1), 1.0)
    cfg = env.unwrapped.cfg

    # Friction: [0.3, 1.3] → [0.1, 1.5] over 15K iters
    sf_min = lerp(*DR_SCHEDULE["static_friction_min"], fraction)
    sf_max = lerp(*DR_SCHEDULE["static_friction_max"], fraction)
    cfg.events.physics_material.params["static_friction_range"] = (sf_min, sf_max)

    # Push: +/-0.5 → +/-1.0 m/s
    push_vel = lerp(*DR_SCHEDULE["push_velocity"], fraction)
    cfg.events.push_robot.params["velocity_range"] = {"x": (-push_vel, push_vel), "y": (-push_vel, push_vel)}

    # External force: +/-3.0 → +/-6.0 N
    ext_force = lerp(*DR_SCHEDULE["ext_force"], fraction)
    cfg.events.base_external_force_torque.params["force_range"] = (-ext_force, ext_force)

    # Mass: +/-5.0 → +/-7.0 kg
    mass_offset = lerp(*DR_SCHEDULE["mass_offset"], fraction)
    cfg.events.add_base_mass.params["mass_distribution_params"] = (-mass_offset, mass_offset)
```

### Why `mode="reset"` is Critical

```python
# In finetune_env_cfg.py:
physics_material = EventTerm(
    func=mdp.randomize_rigid_body_material,
    mode="reset",  # ← Re-randomized EVERY episode reset
    params={"static_friction_range": (0.3, 1.3), ...},
)
```

- **`mode="startup"`**: Randomizes once when the environment is first created. Changes to the config after startup have no effect. This is what the 48hr and 100hr configs use.
- **`mode="reset"`**: Re-randomizes at every episode reset. This lets progressive DR expand the ranges and have them take effect immediately.

If you use `mode="startup"` with progressive DR, the ranges never actually change at runtime — the policy trains with whatever was set at initialization.

---

## 9. Checkpoint Loading Fixes

These fixes were developed across four attempts. Attempt #1 collapsed due to value function mismatch (Fixes 1-4). Attempt #2 collapsed due to noise explosion (Fixes 5-6). Attempt #3 collapsed due to catastrophic forgetting at actor unfreeze (Fixes 7-9). See [TRAINING_METHODOLOGY.md §5.2.7–5.2.12](TRAINING_METHODOLOGY.md) for the academic analysis.

### Fix 1: Actor-Only Loading

The 48hr checkpoint was trained with 14 reward terms. Our environment has 19. Loading the 48hr critic gives wildly wrong value estimates, poisoning advantage computation.

From `train_finetune.py` lines 177–217:

```python
def load_actor_only(runner, checkpoint_path: str):
    """Load ONLY actor weights from checkpoint; critic stays random."""
    loaded_dict = torch.load(checkpoint_path, weights_only=False, map_location=device)
    state_dict = loaded_dict["model_state_dict"]
    policy = runner.alg.policy

    # Load actor MLP weights
    actor_state = {k.replace("actor.", ""): v
                   for k, v in state_dict.items()
                   if k.startswith("actor.")}
    policy.actor.load_state_dict(actor_state, strict=True)

    # Copy noise std from checkpoint (so exploration matches checkpoint's level)
    if "std" in state_dict:
        policy.std.data.copy_(state_dict["std"])

    # Copy actor obs normalizer if present
    norm_state = {k.replace("actor_obs_normalizer.", ""): v
                  for k, v in state_dict.items()
                  if k.startswith("actor_obs_normalizer.")}
    if norm_state and hasattr(policy, 'actor_obs_normalizer'):
        policy.actor_obs_normalizer.load_state_dict(norm_state, strict=False)

    # Critic is intentionally LEFT RANDOM
    critic_keys = [k for k in state_dict if k.startswith("critic.")]
    print(f"  SKIPPED {len(critic_keys)} critic keys (intentionally reset)", flush=True)
```

### Fix 2: Critic Warmup (Actor Freeze + Permanent Std Freeze)

With a random critic, early advantage estimates are garbage. If the actor updates on garbage advantages, it can diverge immediately. Solution: freeze the actor **and noise std** for N iterations while the critic learns the new reward structure.

**Critical detail (learned in Attempt #2):** RSL-RL's `policy.std` is a top-level `nn.Parameter` on `ActorCritic`, NOT inside `policy.actor`. Freezing only `policy.actor.parameters()` leaves `std` trainable. During warmup, the PPO entropy bonus (`-entropy_coef × log(std)`) pushes `std` upward monotonically (since the actor is frozen, there's no surrogate loss to counterbalance it). In Attempt #2, this caused `std` to explode from 0.65 → 5.75+ in 247 iterations.

**Attempt #4 change:** Noise std is now **permanently frozen** (never unfrozen). In Attempt #3, unfreezing std at iter 1000 allowed it to creep from 0.65 → 0.78, adding destructive randomness during the fragile unfreeze transition. With entropy set to 0.0, there is no need for a trainable noise parameter.

```python
def freeze_actor(policy):
    """Freeze actor weights AND noise std so only the critic trains during warmup."""
    for param in policy.actor.parameters():
        param.requires_grad = False
    # CRITICAL: Also freeze noise std — otherwise entropy bonus pushes it up unbounded
    if hasattr(policy, 'std'):
        policy.std.requires_grad = False
    if hasattr(policy, 'log_std'):
        policy.log_std.requires_grad = False

def unfreeze_actor(policy):
    """Unfreeze actor MLP weights only — noise std stays PERMANENTLY FROZEN.

    Attempt #4: std stays at checkpoint's converged 0.65 forever.
    In Attempt #3, unfreezing std allowed entropy bonus to push it from
    0.65 → 0.78, adding destructive randomness during unfreeze transition.
    """
    for param in policy.actor.parameters():
        param.requires_grad = True
    # INTENTIONALLY do NOT unfreeze std/log_std — keep noise permanently at 0.65
```

**LR Warmup at Unfreeze (Attempt #4)** — During warmup, the actor is frozen → KL divergence ≈ 0 → adaptive KL scheduler doubles the learning rate every iteration. After 1000 iterations, the LR would be astronomically inflated. Attempt #3 reset LR to 1e-4 flat, but this was still too aggressive and caused catastrophic forgetting within 30 iterations. Attempt #4 uses a gradual warmup:

```python
# Inside the monkey-patched update function:
_target_lr = agent_cfg.algorithm.learning_rate   # 1e-5
_warmup_start_lr = _target_lr / 5.0              # 2e-6

if not _actor_unfrozen[0] and it >= args_cli.actor_freeze_iters:
    unfreeze_actor(runner.alg.policy)
    _actor_unfrozen[0] = True
    _unfreeze_iter[0] = it

    # Start LR at warmup_start (2e-6), NOT at target (1e-5)
    runner.alg.learning_rate = _warmup_start_lr
    for pg in runner.alg.optimizer.param_groups:
        pg["lr"] = _warmup_start_lr

# Gradual LR ramp: 2e-6 → 1e-5 over lr_warmup_iters
if _actor_unfrozen[0] and _unfreeze_iter[0] is not None:
    iters_since_unfreeze = it - _unfreeze_iter[0]
    if iters_since_unfreeze <= args_cli.lr_warmup_iters:
        warmup_frac = iters_since_unfreeze / max(args_cli.lr_warmup_iters, 1)
        current_lr = _warmup_start_lr + (_target_lr - _warmup_start_lr) * warmup_frac
        runner.alg.learning_rate = current_lr
        for pg in runner.alg.optimizer.param_groups:
            pg["lr"] = current_lr
```

The warmup is applied **both before and after** the PPO update call, overriding any LR adjustment the adaptive KL schedule makes during the ramp period.

### Fix 3: Noise Floor AND Ceiling

- **Attempt #1** — adaptive KL crushed noise from 0.65 → 0.15 (collapse). Fix: `min_std=0.4`.
- **Attempt #2** — entropy bonus pushed noise from 0.65 → 5.75+ (explosion). Fix: `max_std=1.5`.

The clamping now bounds noise in both directions:

```python
def clamp_noise_std(policy, min_std: float, max_std: float = 1.5):
    """Clamp noise std to prevent both exploration collapse AND explosion.

    Attempt #1: adaptive KL crushed noise 0.65 → 0.15 (collapse). Fix: min_std=0.4.
    Attempt #2: entropy bonus pushed noise 0.65 → 5.75+ (explosion). Fix: max_std=1.5.

    The upper bound (1.5) is ~2.3× the checkpoint's converged noise (0.65),
    allowing reasonable exploration growth while preventing runaway.
    """
    with torch.no_grad():
        if hasattr(policy, 'noise_std_type') and policy.noise_std_type == "log":
            log_min = torch.log(torch.tensor(min_std, device=policy.log_std.device))
            log_max = torch.log(torch.tensor(max_std, device=policy.log_std.device))
            policy.log_std.clamp_(min=log_min.item(), max=log_max.item())
        else:
            policy.std.clamp_(min=min_std, max=max_std)
```

Called after every PPO update:

```python
# Inside update_with_fixes():
result = original_update(*args, **kwargs)
clamp_noise_std(runner.alg.policy, args_cli.min_noise_std, args_cli.max_noise_std)
```

CLI arguments:
```
--min_noise_std 0.4    # Floor — prevents exploration collapse
--max_noise_std 1.5    # Ceiling — prevents exploration explosion
```

---

## 10. The Monkey-Patch Pattern

RSL-RL's `OnPolicyRunner.learn()` has no per-iteration callback hooks. To inject progressive DR, critic warmup, and noise floor logic, we replace `runner.alg.update` with a wrapper function.

### Why Monkey-Patching

RSL-RL's training loop is:

```
for iteration in range(max_iterations):
    rollout = collect_rollout()      # ← No hook
    loss = alg.update(rollout)       # ← We patch THIS
    log_metrics()                    # ← No hook
    save_checkpoint()                # ← No hook
```

There is no `on_iteration_start()`, `on_before_update()`, or similar callback. The cleanest injection point is wrapping `alg.update`.

### The Pattern

From `train_finetune.py` lines 402–490:

```python
original_update = runner.alg.update
_iteration_counter = [0]        # Mutable list trick (closures can't rebind nonlocal ints easily)
_actor_unfrozen = [False]
_unfreeze_iter = [None]         # Track when actor was unfrozen for LR warmup
_dr_log_interval = 500
_target_lr = agent_cfg.algorithm.learning_rate   # 1e-5 (final target LR)
_warmup_start_lr = _target_lr / 5.0              # 2e-6 (start of LR warmup)

def update_with_fixes(*args, **kwargs):
    """Wrapper: progressive DR + critic warmup + LR warmup + noise clamp."""
    it = _iteration_counter[0]

    # Fix 2: Unfreeze actor MLP after critic warmup (std stays frozen)
    if not _actor_unfrozen[0] and it >= args_cli.actor_freeze_iters:
        unfreeze_actor(runner.alg.policy)
        _actor_unfrozen[0] = True
        _unfreeze_iter[0] = it
        # Start LR warmup at 2e-6 (not 1e-5 directly — too aggressive)
        runner.alg.learning_rate = _warmup_start_lr
        for pg in runner.alg.optimizer.param_groups:
            pg["lr"] = _warmup_start_lr

    # Fix 9 (Attempt #4): Gradual LR warmup after unfreeze
    if _actor_unfrozen[0] and _unfreeze_iter[0] is not None:
        iters_since_unfreeze = it - _unfreeze_iter[0]
        if iters_since_unfreeze <= args_cli.lr_warmup_iters:
            warmup_frac = iters_since_unfreeze / max(args_cli.lr_warmup_iters, 1)
            current_lr = _warmup_start_lr + (_target_lr - _warmup_start_lr) * warmup_frac
            runner.alg.learning_rate = current_lr
            for pg in runner.alg.optimizer.param_groups:
                pg["lr"] = current_lr

    # Progressive DR
    dr_info = update_dr_params(env, it, args_cli.dr_expansion_iters)

    # Run the actual PPO update
    result = original_update(*args, **kwargs)

    # Fix 3: Clamp noise std after update (safety net — std is frozen but just in case)
    clamp_noise_std(runner.alg.policy, args_cli.min_noise_std, args_cli.max_noise_std)

    # Override adaptive KL's LR during warmup ramp (re-apply our schedule)
    if _actor_unfrozen[0] and _unfreeze_iter[0] is not None:
        iters_since_unfreeze = it - _unfreeze_iter[0]
        if iters_since_unfreeze <= args_cli.lr_warmup_iters:
            warmup_frac = iters_since_unfreeze / max(args_cli.lr_warmup_iters, 1)
            current_lr = _warmup_start_lr + (_target_lr - _warmup_start_lr) * warmup_frac
            runner.alg.learning_rate = current_lr
            for pg in runner.alg.optimizer.param_groups:
                pg["lr"] = current_lr

    _iteration_counter[0] += 1
    return result

runner.alg.update = update_with_fixes    # Replace the method
```

### The `[0]` Mutable List Trick

Python closures can read outer variables, but cannot rebind them without `nonlocal`. Using a mutable list `[0]` lets the inner function modify the value by mutating the container:

```python
_counter = [0]           # List is mutable
def inner():
    _counter[0] += 1     # Mutating the list's element — works without nonlocal
```

This is more concise than using `nonlocal` and works in all Python versions.

---

## 11. Teacher-Student Distillation

### Stage 2a: Teacher Training with Privileged Observations

The teacher receives 254-dim observations (235 standard + 19 privileged):

From `configs/teacher_env_cfg.py`:

```python
@configclass
class SpotTeacherObservationsCfg(SpotFinetuneObservationsCfg):
    @configclass
    class PolicyCfg(SpotFinetuneObservationsCfg.PolicyCfg):
        # Privileged: terrain friction coefficient (from DR)
        terrain_friction = ObsTerm(
            func=mdp.body_physics_material,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="body")},
        )
        # Privileged: per-foot contact forces (clean, no noise)
        foot_contact_forces = ObsTerm(
            func=mdp.contact_forces,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )
```

**Privileged dimension breakdown:**
- Terrain friction: 1 dim (static friction coefficient)
- Foot contact forces: 4 feet × 3 axes = 12 dims
- (Additional terrain info): ~6 dims
- **Total privileged: 19 dims → 235 + 19 = 254 total**

### Weight Surgery: 235 → 254 Dimensions

The Stage 1 checkpoint has a [512, 235] first-layer weight matrix. The teacher needs [512, 254]. We extend it by appending zero-initialized columns:

From `train_teacher.py` lines 67–110:

```python
def extend_checkpoint_for_teacher(checkpoint_path, standard_obs_dim=235):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    modified = {}
    for key, tensor in state_dict.items():
        # Find first layer of actor and critic (input layers)
        # RSL-RL naming: actor.0.weight, critic.0.weight
        if key.endswith(".0.weight") and tensor.shape[1] == standard_obs_dim:
            # Extend: [hidden, 235] → [hidden, 254]
            extra_cols = torch.zeros(tensor.shape[0], 254 - standard_obs_dim)
            modified[key] = torch.cat([tensor, extra_cols], dim=1)
            print(f"  [SURGERY] {key}: {tensor.shape} -> {modified[key].shape}")
        else:
            modified[key] = tensor

    return modified
```

**Why zero-initialized**: The teacher starts with the Stage 1 policy's behavior on standard observations. The privileged dimensions have zero-weight connections, so they initially contribute nothing. The teacher then gradually learns to leverage the privileged information.

### Loading the Extended Checkpoint

```python
# Weight surgery
extended_checkpoint = extend_checkpoint_for_teacher(args_cli.checkpoint)

# Load into the runner's actor_critic (which expects 254-dim input)
runner.alg.actor_critic.load_state_dict(extended_checkpoint["model_state_dict"])
```

### Stage 2b: Student Distillation

The student (235-dim) learns to mimic the teacher (254-dim) via behavior cloning loss combined with PPO:

```
loss = (1 - bc_coef) * PPO_loss + bc_coef * BC_loss
```

Where `BC_loss = MSE(student_action, teacher_action.detach())`

The BC coefficient anneals from 0.8 → 0.2 over training (starts by heavily imitating the teacher, gradually shifts to independent learning).

From `train_distill.py`:

```python
# Load frozen teacher model
teacher_model = ActorCritic(
    num_actor_obs=teacher_obs_dim,     # 254
    num_critic_obs=teacher_obs_dim,
    num_actions=12,
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[512, 256, 128],
    activation="elu",
).to(device)
teacher_model.load_state_dict(teacher_state)
teacher_model.eval()  # Frozen — no gradient updates

# Monkey-patched update with BC coefficient annealing
original_update = runner.alg.update
_iter_counter = [0]

def update_with_distillation(*args, **kwargs):
    it = _iter_counter[0]
    fraction = min(it / max(args_cli.max_iterations, 1), 1.0)
    bc_coef = args_cli.bc_start + (args_cli.bc_end - args_cli.bc_start) * fraction
    # bc_coef: 0.8 → 0.2 over training

    result = original_update(*args, **kwargs)
    _iter_counter[0] += 1
    return result

runner.alg.update = update_with_distillation
```

**DR during distillation**: Fixed at final Stage 1 values (no progressive expansion). The student trains on the hardest conditions from the start:

```python
env_cfg.events.physics_material.params["static_friction_range"] = (0.1, 1.5)
env_cfg.events.push_robot.params["velocity_range"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
env_cfg.events.base_external_force_torque.params["force_range"] = (-6.0, 6.0)
env_cfg.events.add_base_mass.params["mass_distribution_params"] = (-7.0, 7.0)
```

---

## 12. Launch Scripts and Production Ops

### H100 Launch Pattern

From `scripts/train_finetune_h100.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

NUM_ENVS="${NUM_ENVS:-16384}"
MAX_ITERS="${MAX_ITERS:-25000}"
CHECKPOINT="${CHECKPOINT:-/home/t2user/IsaacLab/logs/.../model_27500.pt}"

# Verify checkpoint exists before launching
if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

# Launch training in a detached screen session
screen -dmS finetune bash -c "
    source /home/t2user/miniconda3/etc/profile.d/conda.sh
    conda activate env_isaaclab
    cd /home/t2user/IsaacLab
    export OMNI_KIT_ACCEPT_EULA=YES
    export PYTHONUNBUFFERED=1

    ./isaaclab.sh -p ${TRAIN_SCRIPT} --headless \
        --num_envs ${NUM_ENVS} \
        --max_iterations ${MAX_ITERS} \
        --dr_expansion_iters ${DR_EXPANSION} \
        --actor_freeze_iters ${ACTOR_FREEZE} \
        --lr_warmup_iters ${LR_WARMUP} \
        --min_noise_std ${MIN_NOISE_STD} \
        --max_noise_std ${MAX_NOISE_STD} \
        --checkpoint ${CHECKPOINT} \
        2>&1 | tee -a ${LOG_FILE}
"

# Launch TensorBoard in a separate screen session
screen -dmS tb_finetune bash -c "
    conda activate env_isaaclab
    tensorboard --logdir '${TB_LOGDIR}' --bind_all --port 6006 2>&1
"
```

### Monitoring a Running Job

```bash
# Attach to training output
screen -r finetune

# View log without attaching
tail -f ~/hybrid_st_rl_stage1_*.log

# TensorBoard (from your local machine)
ssh -L 6006:localhost:6006 t2user@ai2ct2
# Then open http://localhost:6006

# List all screen sessions
screen -ls
```

### Local Debug Script

From `scripts/train_local_debug.sh`:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

./isaaclab.sh -p "${TRAIN_SCRIPT}" --headless \
    --num_envs 64 \
    --max_iterations 10 \
    --dr_expansion_iters 5 \
    --actor_freeze_iters 3 \
    --lr_warmup_iters 2 \
    --min_noise_std 0.4 \
    --max_noise_std 1.5 \
    --seed 42 \
    --checkpoint "${CHECKPOINT}"
```

**Post-run validation checklist** (from the script):
- `SKIPPED N critic keys` — actor-only load confirmed
- `Actor MLP UNFROZEN (noise std stays FROZEN)` — actor unfreeze with permanent std lock
- `LR warmup: 2.0e-06 → 1.0e-05` — gradual LR ramp active
- Noise std locked at ~0.65 throughout (permanently frozen)
- `Terrain level > 0` at start — warm start confirmed
- No `NaN` in reward terms
- DR expansion messages appearing

---

## 13. Deployment and Evaluation

### SpotRoughTerrainPolicy — NOT a Scene Object

When deploying the trained policy in standalone Isaac Sim (outside Isaac Lab's `ManagerBasedRLEnv`), the policy wrapper is **not** an Isaac Sim scene prim. Do not call `world.scene.add()` on it.

```python
# WRONG — will crash with AttributeError
world.scene.add(spot_policy)

# CORRECT — just create it, no scene registration
spot = SpotRoughTerrainPolicy(
    name="spot",
    prim_path="/World/Spot",
    position=np.array([0.0, 0.0, 0.6]),  # Use np.array(), NOT Gf.Vec3d()
    checkpoint_path="/path/to/model.pt",
)
```

### Initialization Sequence

```python
# Must happen inside the FIRST physics callback, not at module level
def on_first_step(step_size):
    spot.initialize()      # Creates articulation handles
    spot.post_reset()       # Sets default joint positions
    # NOW the robot is ready to receive actions
```

### Critical Deployment Parameters

| Parameter | Training Value | Deployment Value | Notes |
|-----------|---------------|-----------------|-------|
| Height scan fill | 0.0 (flat) | **0.0** | NOT 1.0 — see Gotchas #7 |
| Action scale | 0.25 | 0.25 | Radians per unit action |
| PD gains | Kp=60, Kd=1.5 | Kp=60, Kd=1.5 | Must match training |
| Physics solver iters | 4 / 0 | **4 / 0** | Training uses GPU PhysX defaults |
| Control frequency | 50 Hz | 50 Hz | decimation=10 at dt=0.002 |

### 4_env_test Evaluation Pipeline

The evaluation runs 8 combinations (2 policies × 4 environments):

```
Policies:  flat_terrain, rough_terrain
Environments: friction, grass, boulder, stairs
→ 8 combinations, 5 episodes each, ~34 min total on H100
```

---

## 14. Gotchas and Pitfalls

Hard-won lessons from development and deployment. **Read this section before making changes.**

### 1. SimulationApp Before omni Imports

```python
# CORRECT — AppLauncher FIRST
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# NOW safe to import omni, pxr, isaaclab modules

# WRONG — segfault
import omni.isaac.core  # SimulationApp doesn't exist yet → crash
```

### 2. `sys.argv` Reset After Parsing

```python
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]  # MUST DO THIS — Gymnasium chokes on unknown args
```

### 3. EULA Acceptance Required

```bash
export OMNI_KIT_ACCEPT_EULA=YES   # Linux/macOS
set OMNI_KIT_ACCEPT_EULA=YES      # Windows CMD
```

Without this, Isaac Sim hangs waiting for interactive EULA acceptance (impossible in headless mode).

### 4. `PYTHONUNBUFFERED=1` + `flush=True`

Isaac Sim buffers stdout aggressively. Without these, you may see no output for minutes:

```bash
export PYTHONUNBUFFERED=1
```

```python
print("Progress...", flush=True)  # Always use flush=True
```

### 5. `nonlocal` in Physics Callbacks

Variables modified inside nested functions (physics callbacks) need `nonlocal`:

```python
drive_mode_idx = 0
def on_physics_step(step_size):
    nonlocal drive_mode_idx   # Without this → UnboundLocalError
    drive_mode_idx += 1
```

### 6. Never Call `simulation_app.close()`

PhysX/CUDA driver teardown deadlocks in the kernel, creating unkillable zombie processes:

```python
# WRONG — creates D-state zombie
simulation_app.close()

# CORRECT — clean exit
import os
os._exit(0)
```

### 7. Height Scan Fill = 0.0 (NOT 1.0)

When deploying without Isaac Lab's RayCaster, fill the 187 height scan dimensions with **0.0**:

```
Training range: [-0.000002, 0.148083], mean: 0.003959
hs=1.0 (wrong): action norm 7.42 → robot collapses
hs=0.0 (correct): action norm 3.08 → robot walks normally
```

### 8. `mode="reset"` vs `mode="startup"` for Events

- `mode="startup"`: Randomized once at env creation. Progressive DR has no effect.
- `mode="reset"`: Re-randomized every episode. Required for progressive DR to work.

### 9. Critic-Reward Mismatch (Actor-Only Loading)

The 48hr checkpoint's critic was trained on 14 reward terms. Our environment has 19. Loading the old critic gives wrong value estimates → poisoned advantages → policy collapse within 2000 iterations.

**Solution**: `load_actor_only()` — skip all critic keys, let critic train from random init.

### 10. Adaptive KL Inflates LR During Actor Freeze

When the actor is frozen, KL divergence ≈ 0 every iteration. The adaptive KL scheduler interprets this as "updates are too conservative" and doubles the LR. After 1000 warmup iterations, LR can reach 0.01+ (vs the intended 1e-4).

**Solution**: Reset LR to `agent_cfg.algorithm.learning_rate` immediately after unfreezing.

### 11. Architecture Must Match Checkpoint Exactly

RSL-RL checkpoints encode the network architecture implicitly in weight shapes. If the config specifies `[512, 256, 128]` but the checkpoint was trained with `[256, 128, 64]`, loading will fail with a shape mismatch error.

```python
# Config MUST match checkpoint:
policy = RslRlPpoActorCriticCfg(
    actor_hidden_dims=[512, 256, 128],   # Must match model_27500.pt
    critic_hidden_dims=[512, 256, 128],  # Must match model_27500.pt
)
```

### 12. GPU PhysX Requires CUDA Tensors

GPU PhysX silently ignores numpy arrays. Use CUDA tensors for `set_joint_efforts()`, `set_joint_positions()`, etc.

### 13. Solver Iteration Matching

Training uses GPU PhysX defaults (4 position iterations, 0 velocity iterations). Deployment must match, or dynamics will differ:

```python
# If deploying with CPU PhysX, set solver iterations to match:
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 0
```

### 14. SpotFlatTerrainPolicy is NOT a Scene Object

Same as SpotRoughTerrainPolicy — no `world.scene.add()`.

### 15. `policy.std` is NOT Inside `policy.actor`

RSL-RL's `ActorCritic` stores the action noise as a top-level `nn.Parameter`:

```
ActorCritic
├── actor   (nn.Sequential)  ← policy.actor.parameters()
├── critic  (nn.Sequential)
├── std     (nn.Parameter)   ← NOT in actor.parameters()!
└── distribution
```

If you freeze `policy.actor.parameters()` to stop the actor from updating, `std` is still trainable. During warmup with a frozen actor, the PPO entropy bonus pushes `std` upward without limit (Attempt #2: 0.65 → 5.75+ in 247 iterations). Always freeze/unfreeze `std` alongside the actor.

### 16. Catastrophic Forgetting at Actor Unfreeze

After critic warmup, the first PPO updates to the newly-unfrozen actor can destroy the pre-trained policy. In Attempt #3, episode length dropped from 206 → 2.5 steps within 30 iterations of unfreeze.

**Root cause:** PPO hyperparameters tuned for training from scratch (LR=1e-4, clip=0.2, 5 epochs) are far too aggressive for fine-tuning. The critic's value predictions become stale as the actor changes, creating a feedback loop.

**Solution:** Ultra-conservative PPO (LR=1e-5, clip=0.1, 3 epochs, entropy=0.0) + gradual LR warmup (2e-6 → 1e-5 over 1000 iterations) + permanently frozen noise std.

### 17. Screen Session Management

Never kill a screen session running Isaac Sim with `screen -X quit`. This leaves zombie processes holding GPU memory. Instead:

```bash
# CORRECT: graceful stop
screen -r finetune    # Attach
Ctrl+C                # Send SIGINT — let the script's signal handler run os._exit(0)

# WRONG: creates unkillable zombies
screen -X quit        # Kills shell, leaves CUDA/PhysX zombies
kill -9 <pid>         # Same problem — D-state zombies survive SIGKILL
```

If zombies already exist, the server may require a reboot.

**18. Fine-Tuning Pre-Trained Policies May Not Work**

Four attempts at fine-tuning the 48hr checkpoint all failed at the freeze/unfreeze boundary. The critic-first warmup approach (freeze actor → train critic → unfreeze actor) creates a fundamental mismatch: the critic learns value estimates for a frozen actor, and even microscopic actor changes (LR=2e-6) cause the estimates to go stale. If fine-tuning collapses despite increasingly conservative hyperparameters, consider training from scratch with terrain curriculum instead. The cost is re-learning basic locomotion (~48hrs), but the training trajectory is smooth and continuous. See `train_from_scratch.py` and `configs/scratch_terrain_cfg.py`.

---

## 15. PPO Hyperparameters

From `configs/finetune_ppo_cfg.py`:

### Runner Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_steps_per_env` | 24 | Rollout length (24 × 0.02s = 0.48s per step) |
| `max_iterations` | 25,000 | ~69 hrs on H100 with 16K envs |
| `save_interval` | 500 | Checkpoint every 500 iters (~1.4 hrs) |
| `experiment_name` | `spot_hybrid_st_rl` | TensorBoard group name |
| `seed` | 42 | Reproducibility |

### Network Architecture

| Parameter | Fine-tune Value | Scratch (Attempt #6) | Rationale |
|-----------|----------------|---------------------|-----------|
| `actor_hidden_dims` | [512, 256, 128] | [512, 256, 128] | Must match 48hr checkpoint / Kumar et al. (2023) |
| `critic_hidden_dims` | [512, 256, 128] | [512, 256, 128] | Must match 48hr checkpoint / Kumar et al. (2023) |
| `activation` | `elu` | `elu` | Standard for locomotion (smooth gradients) |
| `init_noise_std` | 0.65 | **0.5** | Fine-tune: match checkpoint; Scratch: reduced from 1.0 to limit flailing |
| `actor_obs_normalization` | False | **True** | **Critical fix** — required for heterogeneous obs scales (Kumar et al., 2023) |
| `critic_obs_normalization` | False | **True** | **Critical fix** — critic needs clean inputs for accurate value estimates |

**Observation normalization note:** Attempt #5 trained from scratch with
normalization disabled, causing training to stall at iteration 2,000+.
The 235-dim observation vector mixes joint positions (±0.5 rad), joint
velocities (±30 rad/s), and height scan values (±1.0). Without running
mean/std normalization (RSL-RL's `EmpiricalNormalization`), the network's
first layer is dominated by high-magnitude inputs and cannot learn from
the 187-dim height scan. Enabling normalization in Attempt #6 — following
Kumar et al. (2023) — was the single most impactful fix, producing 247s
episode length at iteration 14 vs 6.97s in Attempt #5 at iteration 2,000+.

### PPO Algorithm (Attempt #4 — Ultra-Conservative)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `learning_rate` | **1e-5** | 10× lower than Attempt #3's 1e-4 (prevent catastrophic forgetting) |
| `schedule` | `adaptive` | KL-based LR adjustment (overridden during LR warmup) |
| `desired_kl` | **0.005** | Tighter than Attempt #3's 0.008 |
| `clip_param` | **0.1** | Halved from 0.2 — limits per-update policy change |
| `entropy_coef` | **0.0** | Disabled — noise std is permanently frozen |
| `num_learning_epochs` | **3** | Reduced from 5 — fewer gradient steps per iteration |
| `num_mini_batches` | 8 | 16K envs × 24 steps / 8 = 49K samples per mini-batch |
| `gamma` | 0.99 | Discount factor |
| `lam` | 0.95 | GAE lambda |
| `value_loss_coef` | 1.0 | Critic loss weight |
| `use_clipped_value_loss` | True | Clip value function updates |
| `max_grad_norm` | 1.0 | Gradient clipping |

**Attempt #3 → #4 comparison:**

| Parameter | Attempt #3 | Attempt #4 | Why Changed |
|-----------|-----------|-----------|-------------|
| `learning_rate` | 1e-4 | **1e-5** | 1e-4 caused catastrophic forgetting at unfreeze |
| `clip_param` | 0.2 | **0.1** | 0.2 allowed too-large per-update changes |
| `entropy_coef` | 0.005 | **0.0** | Pushed noise from 0.65→0.78 after unfreeze |
| `num_learning_epochs` | 5 | **3** | 5 × 8 = 40 gradient steps per iter was too many |
| `desired_kl` | 0.008 | **0.005** | Tighter constraint on policy divergence |

### Stage Comparison

| Parameter | 48hr Base | Stage 1 Finetune (#4) | Attempt #5 (Scratch) | **Attempt #6 (Scratch v2)** | Stage 2a (Teacher) | Stage 2b (Distill) |
|-----------|-----------|----------------------|---------------------|---------------------------|--------------------|--------------------|
| LR | 3e-4 | 1e-5 | 1e-3 | **1e-3** | 1e-5 | 5e-6 |
| Clip | 0.2 | 0.1 | 0.2 | **0.2** | 0.1 | 0.1 |
| Entropy | 0.008 | 0.0 | 0.005 | **0.005** | 0.0 | 0.0 |
| Epochs | 5 | 3 | 5 | **5** | 3 | 3 |
| desired_kl | 0.01 | 0.005 | 0.01 | **0.01** | 0.005 | 0.005 |
| init_noise_std | 0.8 | 0.65 (frozen) | 1.0 | **0.5** | 0.65 | 0.65 |
| Obs normalization | ? | False | False | **True** | False | False |
| Obs dims | 235 | 235 | 235 | **235** | 254 | 235 |
| Reward terms | 14 | 19 | 19 | **14 (5 zeroed)** | 19 | 19 |
| Termination | body | body + legs | body + legs | **body only** | body + legs | body + legs |
| Envs | 4,096 | 16,384 | 16,384 | **16,384** | 8,192 | 8,192 |
| Max iters | 27,500 | 25,000 | 15,000 | **15,000** | 20,000 | 10,000 |
| Terrain | ROUGH (6) | ROBUST (12) | SCRATCH (7) | **SCRATCH (7)** | ROBUST (12) | ROBUST (12) |
| Init terrain | level 5 | level 5 | level 0 (flat) | **level 0 (flat)** | level 5 | level 5 |
| Spawn velocity | ? | ±1.5 m/s | ±1.5 m/s | **±0.5 m/s** | ±1.5 m/s | ±1.5 m/s |
| Status | Complete | Failed (critic stale) | Stalled (no normalization) | **Running** | Planned | Planned |

---

## 16. Appendices

### Appendix A: Full Reward Weight Table

| # | Term | Weight | Function | Module |
|---|------|--------|----------|--------|
| 1 | `gait` | +10.0 | `spot_mdp.GaitReward` | isaaclab_tasks |
| 2 | `base_linear_velocity` | +7.0 | `spot_mdp.base_linear_velocity_reward` | isaaclab_tasks |
| 3 | `base_angular_velocity` | +5.0 | `spot_mdp.base_angular_velocity_reward` | isaaclab_tasks |
| 4 | `foot_clearance` | +3.5 | `spot_mdp.foot_clearance_reward` | isaaclab_tasks |
| 5 | `air_time` | +3.0 | `spot_mdp.air_time_reward` | isaaclab_tasks |
| 6 | `velocity_modulation` | +2.0 | `velocity_modulation_reward` | custom |
| 7 | `base_orientation` | -5.0 | `spot_mdp.base_orientation_penalty` | isaaclab_tasks |
| 8 | `base_motion` | -4.0 | `spot_mdp.base_motion_penalty` | isaaclab_tasks |
| 9 | `foot_slip` | -3.0 | `spot_mdp.foot_slip_penalty` | isaaclab_tasks |
| 10 | `action_smoothness` | -2.0 | `spot_mdp.action_smoothness_penalty` | isaaclab_tasks |
| 11 | `body_height_tracking` | -2.0 | `body_height_tracking_penalty` | custom |
| 12 | `stumble` | -2.0 | `stumble_penalty` | custom |
| 13 | `air_time_variance` | -1.0 | `spot_mdp.air_time_variance_penalty` | isaaclab_tasks |
| 14 | `joint_pos` | -1.0 | `spot_mdp.joint_position_penalty` | isaaclab_tasks |
| 15 | `contact_force_smoothness` | -0.5 | `contact_force_smoothness_penalty` | custom |
| 16 | `joint_vel` | -0.05 | `spot_mdp.joint_velocity_penalty` | isaaclab_tasks |
| 17 | `joint_torques` | -0.002 | `spot_mdp.joint_torques_penalty` | isaaclab_tasks |
| 18 | `vegetation_drag` | -0.001 | `VegetationDragReward` | custom (class) |
| 19 | `joint_acc` | -0.0005 | `spot_mdp.joint_acceleration_penalty` | isaaclab_tasks |

### Appendix B: DR Schedule Timeline

| Iteration | Fraction | Static Friction | Push Vel | Ext Force | Mass Offset |
|-----------|----------|----------------|----------|-----------|-------------|
| 0 | 0% | [0.30, 1.30] | ±0.50 m/s | ±3.0 N | ±5.0 kg |
| 5,000 | 33% | [0.23, 1.37] | ±0.67 m/s | ±4.0 N | ±5.7 kg |
| 10,000 | 67% | [0.17, 1.43] | ±0.83 m/s | ±5.0 N | ±6.3 kg |
| 15,000 | 100% | [0.10, 1.50] | ±1.00 m/s | ±6.0 N | ±7.0 kg |
| 25,000 | 100% | [0.10, 1.50] | ±1.00 m/s | ±6.0 N | ±7.0 kg |

### Appendix C: 235-Dim Observation Breakdown

| Component | Dims | Range/Noise | Source Function |
|-----------|------|-------------|-----------------|
| Base linear velocity | 3 | ±0.15 noise | `mdp.base_lin_vel` |
| Base angular velocity | 3 | ±0.15 noise | `mdp.base_ang_vel` |
| Projected gravity | 3 | ±0.05 noise | `mdp.projected_gravity` |
| Velocity commands | 3 | No noise | `mdp.generated_commands` |
| Joint positions (relative) | 12 | ±0.05 noise | `mdp.joint_pos_rel` |
| Joint velocities (relative) | 12 | ±0.5 noise | `mdp.joint_vel_rel` |
| Last actions | 12 | No noise | `mdp.last_action` |
| **Proprioceptive subtotal** | **48** | | |
| Height scan (17×11 grid) | 187 | ±0.15 noise, clip [-1,1] | `mdp.height_scan` |
| **Total** | **235** | | |

### Appendix D: API Reference Table

| Import | Class/Function | Usage |
|--------|---------------|-------|
| `isaaclab.app` | `AppLauncher` | Creates SimulationApp, parses --headless etc. |
| `isaaclab.utils` | `configclass` | Decorator for Isaac Lab configuration dataclasses |
| `isaaclab.envs` | `ManagerBasedRLEnv` | Base environment class (Gymnasium entry_point) |
| `isaaclab.managers` | `ObservationTermCfg` | Defines one observation term |
| `isaaclab.managers` | `RewardTermCfg` | Defines one reward term |
| `isaaclab.managers` | `EventTermCfg` | Defines one DR/reset event |
| `isaaclab.managers` | `CurriculumTermCfg` | Defines curriculum progression rule |
| `isaaclab.managers` | `TerminationTermCfg` | Defines episode termination condition |
| `isaaclab.managers` | `SceneEntityCfg` | References a scene entity by name |
| `isaaclab.managers` | `ManagerTermBase` | Base class for class-based reward terms |
| `isaaclab.sensors` | `RayCasterCfg` | Height scanner configuration |
| `isaaclab.terrains` | `TerrainImporterCfg` | Terrain mesh generation/import |
| `isaaclab.terrains` | `TerrainGeneratorCfg` | Procedural terrain generator config |
| `isaaclab.utils.io` | `dump_yaml` | Serialize config to YAML file |
| `isaaclab.utils.noise` | `AdditiveUniformNoiseCfg` | Uniform observation noise |
| `isaaclab_rl.rsl_rl` | `RslRlVecEnvWrapper` | Wraps Isaac Lab env for RSL-RL |
| `isaaclab_rl.rsl_rl` | `RslRlOnPolicyRunnerCfg` | PPO runner configuration |
| `isaaclab_rl.rsl_rl` | `RslRlPpoActorCriticCfg` | Network architecture config |
| `isaaclab_rl.rsl_rl` | `RslRlPpoAlgorithmCfg` | PPO hyperparameters config |
| `isaaclab_assets.robots.spot` | `SPOT_CFG` | Pre-defined Spot robot URDF config |
| `rsl_rl.runners` | `OnPolicyRunner` | PPO training loop (rollout + update + log) |
| `rsl_rl.modules` | `ActorCritic` | Actor-critic network module |
| `isaaclab_tasks...spot.mdp` | `spot_mdp` | Spot-specific reward/observation functions |
| `isaaclab_tasks...velocity.mdp` | `mdp` | Generic locomotion reward/observation functions |

---

*Last updated: February 25, 2026 — Attempt #6 from-scratch training with observation normalization fix informed by Kumar et al. (2023). Attempt #5 stalled due to disabled obs normalization, aggressive termination, and high init_noise_std. Attempts #1–4 fine-tuning all failed at freeze/unfreeze boundary.*
*AI2C Tech Capstone — Hybrid ST-RL Training Pipeline*
