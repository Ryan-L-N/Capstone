# RT Policy Training Package for ARL
## Boston Dynamics Spot — Rough Terrain Locomotion via PPO

**Project:** AI2C Tech Capstone — MS for Autonomy
**Date:** February 15, 2026
**POC:** AI2C Capstone Team

---

## 1. What This Is

This package contains everything needed to train a **reinforcement learning locomotion policy** for the Boston Dynamics Spot quadruped robot on rough terrain (stairs, slopes, rubble, boxes). The policy is trained using **Proximal Policy Optimization (PPO)** inside NVIDIA Isaac Lab's GPU-accelerated simulation environment.

The trained policy takes 235 sensor observations as input and outputs 12 joint position targets (3 joints per leg x 4 legs) at 50 Hz, enabling Spot to walk, trot, climb stairs, and traverse uneven terrain autonomously.

### Why This Training Exists

Our initial 5,000-iteration training run (~45 minutes on an RTX 4090) produced a policy that could barely walk and collapsed immediately when deployed in a standalone simulation. Root cause analysis identified three problems:

1. **Undertrained** — 5,000 iterations is insufficient for a 235-dimensional observation space with terrain curriculum
2. **Poor proprioception** — the policy never learned to use joint state for balance recovery
3. **Inefficient joint actuation** — high torques, jerky motions, no smoothness incentive

This package implements a **30,000-iteration training run** (~48 hours on an H100, scalable to RTX hardware) with tuned reward weights that specifically target proprioception, joint efficiency, and locomotion quality.

---

## 2. Training Status — COMPLETE (H100 NVL)

Training completed on February 16, 2026 on an H100 NVL (96 GB VRAM) server. 30,000 / 30,000 iterations.

| Metric | Start (Iter 0) | Final (Iter 29,996) | Change |
|--------|---------------:|--------------------:|--------|
| Mean Reward | -0.90 | +143.74 | +144.64 |
| Episode Length | 20 steps (0.4s) | 573 steps (11.5s) | 28.6x |
| Gait Reward | 0.06 | 5.28 | 88x |
| Terrain Level | 3.18 | 4.42 | +1.24 |
| Body Contact (falls) | 22% | 57.5% | stabilized |
| Timeout (survived) | 0.9% | 42.2% | 47x |
| Throughput | — | ~30,000 steps/s | 8,192 parallel envs |

The policy progressed through three training phases:
- **Phase 1 (0–10k):** Learned to stand, walk, and follow velocity commands
- **Phase 2 (10k–20k):** Mastered rough terrain — stairs, slopes, obstacles
- **Phase 3 (20k–30k):** Polished gait quality, energy efficiency, perturbation recovery

### Deployment Status — WORKING

The trained policy has been successfully deployed in a standalone obstacle course (100m, 12 terrain segments) with WASD + Xbox controller teleop. Dual gait switching between NVIDIA's flat terrain policy and our trained rough terrain policy works via G key / RB button.

**Critical deployment fix**: Height scan fill value must be **0.0** (not 1.0). See Section 12 below.

---

## 3. Software Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| **NVIDIA Isaac Sim** | 5.1.0 | Omniverse-based robotics simulator |
| **NVIDIA Isaac Lab** | 2.3.0 | RL framework on top of Isaac Sim |
| **RSL-RL** | (bundled with Isaac Lab) | PPO implementation from RSL @ ETH Zurich |
| **Python** | 3.11.x | Isaac Lab 2.3.0 requirement |
| **PyTorch** | 2.x with CUDA | GPU-accelerated training |
| **CUDA** | 12.x | GPU compute |

### Isaac Lab Installation

Isaac Lab must be installed following the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). The conda environment should have Isaac Sim, Isaac Lab, and RSL-RL available as Python packages.

---

## 4. Hardware Requirements

| GPU | Recommended Envs | Est. Steps/s | Est. Total Time | VRAM Usage |
|-----|-------------------|--------------|-----------------|------------|
| **H100 NVL (96 GB)** | 8,192 | ~36,000 | ~30 hours | ~10 GB |
| **RTX 6000 Ada (48 GB)** | 4,096–8,192 | ~18,000–25,000 | ~40–55 hours | ~8–12 GB |
| **RTX 4090 (24 GB)** | 2,048–4,096 | ~10,000–16,000 | ~55–80 hours | ~6–10 GB |
| **RTX A6000 (48 GB)** | 4,096–8,192 | ~15,000–22,000 | ~45–65 hours | ~8–12 GB |
| **RTX 2000 Ada (16 GB)** | 1,024–2,048 | ~4,000–6,000 | ~23 days | ~5–8 GB |

**Minimum:** Any NVIDIA GPU with 16+ GB VRAM and CUDA 12.x support.
**Recommended:** 24+ GB VRAM for reasonable training times (under 4 days).

The number of parallel environments (`--num_envs`) should be scaled to the GPU's VRAM. More environments = higher throughput = faster training, but more memory.

---

## 5. File Manifest

```
RT_POLICY_FOR_ARL/
|
|-- README.md                          <-- This document
|-- spot_rough_48h_cfg.py              <-- Main training script (standalone)
|-- train_spot_rough_48h.sh            <-- Shell launcher script
|-- debug_10iter.sh                    <-- 10-iteration debug run (verify setup)
|-- eval_checkpoints.sh                <-- Post-training checkpoint evaluation
|-- TRAINING_PLAN.md                   <-- Detailed training plan & rationale
|-- LESSONS_LEARNED.md                 <-- Setup issues & fixes from H100 deployment
|
|-- isaac_lab_spot_configs/            <-- Spot-specific Isaac Lab configs
    |-- __init__.py                    <-- Gym environment registrations
    |-- rough_env_cfg.py               <-- Spot rough terrain environment config
    |-- agents/
    |   |-- __init__.py
    |   |-- rsl_rl_ppo_cfg.py          <-- Base PPO runner configs (Flat + Rough)
    |-- mdp/
        |-- __init__.py
        |-- rewards.py                 <-- 14 Spot-specific reward functions
        |-- events.py                  <-- Domain randomization (joint reset)
```

---

## 6. Installation on Target Server

### Step 1: Install Isaac Lab

Follow the official Isaac Lab installation. Ensure the conda environment has `isaacsim`, `isaaclab`, and `rsl_rl` available:

```bash
conda activate <your_isaaclab_env>
python -c "import isaacsim; print('Isaac Sim: OK')"
python -c "import isaaclab; print(f'Isaac Lab: {isaaclab.__version__}')"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Step 2: Install Spot Rough Terrain Configs

The Spot rough terrain configs may not be present in your Isaac Lab installation (they ship with Isaac Lab 2.3.0 but some installations only have the flat config). Copy the provided configs:

```bash
# Find your Isaac Lab task configs directory
SPOT_DIR=$(python -c "import isaaclab_tasks; import os; print(os.path.join(os.path.dirname(isaaclab_tasks.__file__), 'manager_based', 'locomotion', 'velocity', 'config', 'spot'))")

echo "Spot config directory: $SPOT_DIR"

# Check if rough_env_cfg.py already exists
if [ -f "$SPOT_DIR/rough_env_cfg.py" ]; then
    echo "rough_env_cfg.py already exists — skipping"
else
    echo "Installing Spot rough terrain configs..."
    cp isaac_lab_spot_configs/rough_env_cfg.py "$SPOT_DIR/"
    cp isaac_lab_spot_configs/__init__.py "$SPOT_DIR/"
    cp isaac_lab_spot_configs/agents/rsl_rl_ppo_cfg.py "$SPOT_DIR/agents/"
    # mdp/ files (rewards.py, events.py) should already exist
    # but copy them if missing:
    cp isaac_lab_spot_configs/mdp/rewards.py "$SPOT_DIR/mdp/" 2>/dev/null
    cp isaac_lab_spot_configs/mdp/events.py "$SPOT_DIR/mdp/" 2>/dev/null
    echo "Done."
fi
```

### Step 3: Accept EULA (Headless Servers)

For non-interactive / headless training, set the EULA environment variable:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

Add this to your `.bashrc` or the training script to make it permanent.

### Step 4: Fix Line Endings (If Transferring from Windows)

Shell scripts may have Windows line endings (`\r\n`) which break on Linux:

```bash
sed -i "s/\r$//" *.sh
```

---

## 7. Running the Training

### Quick Start (Recommended)

```bash
# 1. Start a screen/tmux session (training takes hours/days)
screen -S spot_training

# 2. Activate your Isaac Lab conda environment
conda activate <your_isaaclab_env>
export OMNI_KIT_ACCEPT_EULA=YES

# 3. Run the debug test first (10 iterations, ~2 minutes)
cd /path/to/RT_POLICY_FOR_ARL
bash debug_10iter.sh

# 4. If debug succeeds, launch full training
cd ~/IsaacLab  # Must run from IsaacLab root
./isaaclab.sh -p /path/to/RT_POLICY_FOR_ARL/spot_rough_48h_cfg.py \
    --headless \
    --num_envs 4096 \
    --max_iterations 30000

# 5. Detach screen: Ctrl+A, then D
# 6. Reconnect later: screen -r spot_training
```

### Adjusting for Your Hardware

The `--num_envs` flag controls how many parallel simulation environments run on the GPU. Scale it to your VRAM:

| VRAM | Recommended `--num_envs` |
|------|--------------------------|
| 16 GB | 1,024–2,048 |
| 24 GB | 2,048–4,096 |
| 48 GB | 4,096–8,192 |
| 80+ GB | 8,192 |

You can also adjust `--max_iterations`. The policy shows meaningful locomotion by iteration ~2,500 and solid rough terrain performance by ~15,000. 30,000 is the full polish run.

### Resuming from Checkpoint

If training is interrupted, resume from the latest checkpoint:

```bash
./isaaclab.sh -p /path/to/spot_rough_48h_cfg.py \
    --headless \
    --num_envs 4096 \
    --resume
```

---

## 8. Technical Architecture

### 8.1 Observation Space (235 dimensions)

The policy receives a 235-dimensional observation vector every control step (50 Hz):

| Observation | Dimensions | Description |
|-------------|-----------|-------------|
| `base_lin_vel` | 3 | Base linear velocity in body frame (m/s) |
| `base_ang_vel` | 3 | Base angular velocity in body frame (rad/s) |
| `projected_gravity` | 3 | Gravity vector projected into body frame |
| `velocity_commands` | 3 | Target velocity commands [vx, vy, yaw_rate] |
| `joint_pos` | 12 | Joint positions relative to default (rad) |
| `joint_vel` | 12 | Joint velocities relative to default (rad/s) |
| `actions` | 12 | Previous action (joint position targets) |
| `height_scan` | **187** | Terrain height measurements (17x11 grid, 0.1m resolution, 1.6m x 1.0m area) |
| **Total** | **235** | |

**Important:** The height scan uses a `GridPattern(resolution=0.1, size=[1.6, 1.0])` which produces a 17x11 = 187 point grid (endpoints included). This is NOT 160 (16x10) as some documentation may state.

### 8.2 Action Space (12 dimensions)

The policy outputs 12 joint position offsets, scaled by 0.25 and added to the default joint positions. These are sent to PD position controllers at each joint:

- **Kp (stiffness):** 60.0
- **Kd (damping):** 1.5
- **Action scale:** 0.25
- **Control frequency:** 50 Hz (physics at 500 Hz, decimation = 10)

Joint ordering follows the Spot URDF: `[fl_hx, fl_hy, fl_kn, fr_hx, fr_hy, fr_kn, hl_hx, hl_hy, hl_kn, hr_hx, hr_hy, hr_kn]`

### 8.3 Network Architecture

```
Actor:  235 --> [512] --> [256] --> [128] --> 12   (ELU activations)
Critic: 235 --> [512] --> [256] --> [128] --> 1    (ELU activations)
```

This is the standard architecture for quadruped locomotion (used by ANYmal-C/D, Unitree Go2, etc.). The checkpoint `.pt` files contain both actor and critic weights.

### 8.4 Reward Function (14 Terms)

The reward function has 5 positive (task) terms and 9 negative (penalty) terms:

**Positive (task):**

| Term | Weight | Function |
|------|--------|----------|
| `gait` | +10.0 | Trot enforcer — synchronizes diagonal foot pairs |
| `base_linear_velocity` | +7.0 | Exponential kernel tracking of commanded XY velocity |
| `base_angular_velocity` | +5.0 | Exponential kernel tracking of commanded yaw rate |
| `air_time` | +5.0 | Rewards appropriate swing/stance phase timing |
| `foot_clearance` | +2.5 | Rewards foot height during swing phase (for stairs) |

**Negative (penalties):**

| Term | Weight | Function |
|------|--------|----------|
| `base_orientation` | -5.0 | Penalizes roll/pitch deviation from upright |
| `base_motion` | -3.0 | Penalizes vertical bouncing and lateral sway |
| `action_smoothness` | -2.0 | Penalizes jerky action changes between timesteps |
| `foot_slip` | -1.0 | Penalizes foot sliding while in ground contact |
| `joint_pos` | -1.0 | Penalizes deviation from default joint positions |
| `air_time_variance` | -1.0 | Penalizes asymmetric gaits across legs |
| `joint_vel` | -0.02 | Penalizes high joint velocities |
| `joint_torques` | -0.002 | Penalizes high joint torques (energy efficiency) |
| `joint_acc` | -0.0005 | Penalizes joint acceleration (smooth trajectories) |

These weights are **overrides** applied on top of the base `SpotRoughRewardsCfg` defaults by the `apply_reward_overrides()` function in `spot_rough_48h_cfg.py`. The reward functions themselves (in `mdp/rewards.py`) are unchanged.

### 8.5 PPO Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 3e-4 | Adaptive schedule (KL-based) |
| Desired KL | 0.01 | Target KL divergence for LR adjustment |
| Entropy coefficient | 0.008 | Exploration incentive |
| Discount (gamma) | 0.99 | |
| GAE lambda | 0.95 | |
| PPO clip | 0.2 | |
| Mini-batches | 8 | Per update epoch |
| Learning epochs | 5 | Per iteration |
| Steps per env | 24 | Rollout length |
| Max grad norm | 1.0 | Gradient clipping |
| Initial noise std | 0.8 | Action exploration noise |
| Observation normalization | **Disabled** | Important for deployment |

### 8.6 Domain Randomization

| Event | Range | Mode |
|-------|-------|------|
| Physics material (friction) | Static: [0.5, 1.25], Dynamic: [0.4, 1.0] | Startup |
| Base mass perturbation | [-5.0, +5.0] kg | Startup |
| Joint reset around default | Position: +/-0.2 rad, Velocity: +/-2.5 rad/s | Reset |
| Base pose reset | +/-0.5m XY, full yaw range | Reset |
| External force perturbation | +/-3.0 N force, +/-1.0 Nm torque | Reset |
| Push perturbation | +/-0.5 m/s XY velocity, every 10-15s | Interval |

### 8.7 Terrain Curriculum

Training uses NVIDIA's `ROUGH_TERRAINS_CFG` which includes 6 terrain types:
- Flat ground
- Random uniform terrain
- Slope terrain (ascending/descending)
- Stair terrain (ascending/descending)
- Pyramid stairs
- Discrete obstacles (boxes)

The curriculum automatically adjusts terrain difficulty based on robot performance — robots that survive longer episodes are moved to harder terrain levels.

---

## 9. Monitoring Training

### TensorBoard

```bash
# In a separate terminal/screen window on the training server:
tensorboard --logdir ~/IsaacLab/logs/rsl_rl/spot_rough --port 6006 --bind_all
```

Access via browser at `http://<server_ip>:6006`.

### Key Metrics to Watch

| Metric | Healthy Range | Red Flag |
|--------|--------------|----------|
| `mean_reward` | Increasing trend | Flatline after 5,000 iterations |
| `mean_episode_length` | Growing toward 1000 (20s) | Dropping below 200 (4s) |
| `terrain_levels` | Increasing 0 → 4+ | Stuck at 0–1 after 10k iterations |
| `Episode_Termination/time_out` | Growing above 30% | Stuck below 10% |
| `Episode_Termination/body_contact` | Decreasing below 70% | Stuck above 90% after 5k |
| `learning_rate` | 1e-4 to 3e-4 | Drops below 1e-6 |
| `policy_loss` | Small, stable | NaN or diverging |

### Training Log

The training script outputs iteration summaries to stdout. If using `tee`:

```bash
./isaaclab.sh -p /path/to/spot_rough_48h_cfg.py --headless --num_envs 4096 2>&1 | tee training.log
```

---

## 10. After Training

### Evaluate Checkpoints

```bash
bash eval_checkpoints.sh <run_directory_name>
# Example: bash eval_checkpoints.sh 2026-02-15_10-00-00_48h_proprioception
```

This evaluates checkpoints at every 5,000 iterations on all terrain levels to find the best model.

### Retrieve the Best Checkpoint

Checkpoints are saved to: `~/IsaacLab/logs/rsl_rl/spot_rough/<timestamp>_48h_proprioception/`

Each checkpoint is a `.pt` file containing the full actor-critic state dict. Transfer the best one:

```bash
scp user@server:~/IsaacLab/logs/rsl_rl/spot_rough/<run>/model_<iter>.pt ./best_policy.pt
```

### Checkpoint Contents

Each `model_XXXXX.pt` file contains:
- Actor network weights (235 -> 512 -> 256 -> 128 -> 12)
- Critic network weights (235 -> 512 -> 256 -> 128 -> 1)
- Optimizer state
- Training iteration count

---

## 11. Known Issues & Fixes

These were discovered during our H100 deployment. See `LESSONS_LEARNED.md` for full details.

| Issue | Symptom | Fix |
|-------|---------|-----|
| **CRLF line endings** | `\r: command not found` when running .sh files | `sed -i "s/\r$//" *.sh` |
| **Conda not in PATH** | `conda: command not found` via SSH | Add `eval "$(/path/to/miniconda3/bin/conda shell.bash hook)"` to script |
| **EULA prompt** | `import isaacsim` hangs waiting for input | `export OMNI_KIT_ACCEPT_EULA=YES` |
| **Missing rough configs** | `ModuleNotFoundError: rough_env_cfg` | Install configs from `isaac_lab_spot_configs/` (Step 2 above) |
| **Height scan fill value** | Robot falls instantly if filled with 1.0 | Use 0.0 for flat ground — actual training values are ~0.0, NOT 1.0 |

---

## 12. Deployment Notes

After training, the policy `.pt` file can be loaded into a standalone Isaac Sim application (no Isaac Lab required) for deployment testing. Critical requirements for the deployment wrapper:

1. **Observation vector must be exactly 235 dimensions** in the exact order listed in Section 8.1
2. **Physics solver:** GPU PhysX with 4 position / 0 velocity solver iterations
3. **PD controller gains:** Kp=60, Kd=1.5 (must match training)
4. **Action scale:** 0.25 (must match training)
5. **No observation normalization** — `actor_obs_normalization=False`
6. **All tensors on CUDA** — ArticulationView setters require `torch.Tensor` on `cuda:0`, numpy arrays silently fail
7. **Control frequency:** 50 Hz (one inference per 10 physics steps at 500 Hz)
8. **Quaternion convention:** Isaac Sim uses [w, x, y, z] (scalar-first)
9. **Height scan fill value: 0.0** (NOT 1.0) — on flat ground, training height_scan ≈ 0.0

### CRITICAL: Height Scan Fill Value

When deploying without a raycaster, the 187 height_scan dimensions must be filled with **0.0** (flat ground assumption). Do NOT use 1.0.

**Why this matters:** The trained actor is extremely sensitive to height_scan values:
- `hs = 0.0` → action norm 3.08 (normal walking)
- `hs = 1.0` → action norm 7.42 (catastrophic — robot falls instantly)

The original deployment code used 1.0 based on incorrect source code analysis. Actual runtime values from the training environment show height_scan ≈ 0.0 on flat ground.

### GPU PhysX Requirement

The standalone deployment MUST use GPU PhysX (`backend="torch"`, `device="cuda:0"`). Policies trained with GPU PhysX produce different dynamics than CPU PhysX, causing divergence and falls.

Use `NumpyRobotWrapper` to bridge between GPU PhysX (CUDA tensors) and the robot API (numpy):
```python
class NumpyRobotWrapper:
    def get_joint_positions(self):
        t = self._av.get_joint_positions()
        return t.cpu().numpy()[0] if isinstance(t, torch.Tensor) else np.array(t).flatten()
```

---

## 13. Contact

For questions about this training package, contact the AI2C Tech Capstone team.

**Framework references:**
- [NVIDIA Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [RSL-RL (ETH Zurich)](https://github.com/leggedrobotics/rsl_rl)
- [Isaac Sim 5.1.0](https://developer.nvidia.com/isaac-sim)
