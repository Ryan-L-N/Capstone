# 48-Hour Spot Rough Terrain Training Plan (H100 NVL)

**Date:** February 13, 2026
**Hardware:** NVIDIA H100 NVL (96 GB VRAM)
**Server:** `172.24.254.24` (ai2ct2), login `t2user`
**Conda env:** `env_isaaclab`
**Task:** `Isaac-Velocity-Rough-Spot-v0`
**Framework:** RSL-RL (PPO)

---

## 1. Problem Statement

The previous rough terrain training (5,000 iterations, ~45 min on RTX 4090) produced a policy
that barely moves and collapses when deployed standalone. Root causes:

1. **Undertrained** - 5,000 iterations is far too few for a 235-dim observation space
2. **Poor proprioception** - the policy never learned to use joint state for balance recovery
3. **Inefficient joint actuation** - high torques, jerky motions, no smoothness incentive tuning
4. **Sim-to-sim gap** - training physics must be exactly replicated at deployment

This plan describes a 48-hour training run on the H100 that targets all three weaknesses.

---

## 2. Hardware Budget

From the H100 stress test (ANYmal-C benchmarks, same architecture):

| Envs  | Steps/s | GPU Temp | VRAM  | Physics | Notes                      |
|-------|---------|----------|-------|---------|----------------------------|
| 4,096 | 22,600  | 45°C     | 8 GB  | Clean   | Conservative               |
| 8,192 | 36,000  | 49°C     | 10 GB | Clean   | **Optimal (our target)**   |
| 16,384| 54,000  | 56°C     | 16 GB | Clean   | Good but hotter            |
| 32,768| 72,000  | 64°C     | 29 GB | Marginal| Occasional instability     |

**Selected: 8,192 envs** - best throughput-per-watt, clean physics, plenty of VRAM headroom.

### Iteration budget

At 36,000 steps/s with `num_steps_per_env=24` and 8,192 envs:

- Steps per iteration: 24 x 8,192 = 196,608
- Time per iteration: 196,608 / 36,000 = ~5.5 seconds
- Iterations per hour: ~655
- **48 hours = ~31,400 iterations**

We set `max_iterations = 30,000` with `save_interval = 500` (60 checkpoints).

---

## 3. Training Phases

The 30,000-iteration run naturally breaks into three phases:

### Phase 1: Foundation (iterations 0 - 10,000)
**Goal:** Learn to stand, walk, maintain balance, follow velocity commands

- Terrain curriculum starts at level 0 (flat/gentle)
- High entropy encourages exploration of gait space
- Gait reward (weight 10.0) dominates early training to establish trot
- LR starts at 3e-4, adaptive schedule keeps KL near 0.01

### Phase 2: Rough Terrain Mastery (iterations 10,000 - 20,000)
**Goal:** Climb stairs, traverse slopes, navigate obstacles

- Terrain curriculum progresses to harder levels
- Foot clearance reward (weight 2.5) drives high stepping
- Height scan observations become critical for terrain reading
- Joint efficiency penalties prevent brute-force solutions

### Phase 3: Robustness & Efficiency (iterations 20,000 - 30,000)
**Goal:** Polish gait quality, reduce energy, handle perturbations

- Most environments on difficult terrain
- Joint torque/acceleration penalties shape efficient actuation
- Action smoothness penalty eliminates jitter
- Push perturbations test recovery

---

## 4. Reward Engineering

### 4.1 Current Rewards (Baseline)

From `rough_env_cfg.py` — these are the Spot-specific rewards, NOT the base mdp rewards:

| Reward Term           | Weight  | Function                          |
|-----------------------|---------|-----------------------------------|
| air_time              | +5.0    | Rewards longer swing/stance phases|
| base_angular_velocity | +5.0    | Exponential kernel yaw tracking   |
| base_linear_velocity  | +5.0    | Exponential kernel + ramp scaling |
| foot_clearance        | +2.0    | Rewards foot height during swing  |
| gait                  | +10.0   | Trot enforcer (sync/async pairs)  |
| action_smoothness     | -1.0    | L2 diff between consecutive acts  |
| air_time_variance     | -1.0    | Penalizes asymmetric gaits        |
| base_motion           | -2.0    | Penalizes vertical/lateral drift  |
| base_orientation      | -3.0    | Penalizes roll/pitch deviation    |
| foot_slip             | -0.5    | Penalizes feet sliding on ground  |
| joint_acc             | -1e-4   | Penalizes joint acceleration      |
| joint_pos             | -0.7    | Penalizes deviation from default  |
| joint_torques         | -5e-4   | Penalizes high torques            |
| joint_vel             | -1e-2   | Penalizes joint velocities        |

### 4.2 Modified Rewards for 48h Training

Changes target three areas: **proprioception**, **joint efficiency**, and **locomotion quality**.

| Reward Term           | Old     | **New** | Rationale                                        |
|-----------------------|---------|---------|--------------------------------------------------|
| air_time              | +5.0    | +5.0    | Keep — already well-tuned                        |
| base_angular_velocity | +5.0    | +5.0    | Keep — yaw tracking is solid                     |
| base_linear_velocity  | +5.0    | +7.0    | **Increase** — stronger velocity tracking signal |
| foot_clearance        | +2.0    | +2.5    | **Increase** — higher stepping for stairs        |
| gait                  | +10.0   | +10.0   | Keep — trot enforcement is critical              |
| action_smoothness     | -1.0    | **-2.0**| **Increase** — smoother actuation                |
| air_time_variance     | -1.0    | -1.0    | Keep                                             |
| base_motion           | -2.0    | **-3.0**| **Increase** — less vertical bouncing            |
| base_orientation      | -3.0    | **-5.0**| **Increase** — stronger upright incentive        |
| foot_slip             | -0.5    | **-1.0**| **Increase** — less foot sliding                 |
| joint_acc             | -1e-4   | **-5e-4**| **5x increase** — smoother joint trajectories   |
| joint_pos             | -0.7    | **-1.0**| **Increase** — stay closer to default stance     |
| joint_torques         | -5e-4   | **-2e-3**| **4x increase** — energy efficiency             |
| joint_vel             | -1e-2   | **-2e-2**| **2x increase** — slower, controlled movements  |

**Key philosophy:** The original training was too permissive on penalties. With 30,000 iterations
(6x more than before), the policy has enough time to find efficient solutions under tighter
constraints. Stronger penalties early → cleaner gaits that survive deployment.

---

## 5. Network Architecture

### Previous (5k run)
```
Actor:  [512, 256, 128] (ELU)  — 235 obs → 12 actions
Critic: [512, 256, 128] (ELU)  — 235 obs → 1 value
```

### 48h Training
```
Actor:  [512, 256, 128] (ELU)  — 235 obs → 12 actions  (KEEP)
Critic: [512, 256, 128] (ELU)  — 235 obs → 1 value     (KEEP)
```

**Rationale:** The [512, 256, 128] architecture is standard for quadruped locomotion (used by
ANYmal-C/D, Go2, etc.). The problem was iteration count, not model capacity. Keeping the same
architecture also means checkpoints are compatible for warm-starting if needed.

---

## 6. PPO Hyperparameters

| Parameter            | Previous | **48h**   | Rationale                                |
|----------------------|----------|-----------|------------------------------------------|
| max_iterations       | 5,000    | **30,000**| 48h budget at 8,192 envs                 |
| num_steps_per_env    | 24       | 24        | Standard for locomotion                  |
| num_envs             | 4,096    | **8,192** | H100 optimal throughput                  |
| save_interval        | 100      | **500**   | 60 checkpoints over 30k iterations       |
| learning_rate        | 1e-3     | **3e-4**  | Lower for stability over long training   |
| schedule             | adaptive | adaptive  | KL-based LR adjustment                   |
| desired_kl           | 0.01     | 0.01      | Standard target                          |
| gamma                | 0.99     | 0.99      | Standard discount                        |
| lam                  | 0.95     | 0.95      | GAE lambda                               |
| clip_param           | 0.2      | 0.2       | PPO clip range                           |
| entropy_coef         | 0.005    | **0.008** | More exploration for longer training     |
| value_loss_coef      | 1.0      | 1.0       | Keep — value function learning rate      |
| num_learning_epochs  | 5        | 5         | Standard for PPO                         |
| num_mini_batches     | 4        | **8**     | Double for 8,192 envs (batch=24k/batch)  |
| max_grad_norm        | 1.0      | 1.0       | Gradient clipping                        |
| init_noise_std       | 1.0      | **0.8**   | Slightly less initial exploration noise  |
| actor_obs_norm       | False    | False     | Spot-specific (no obs normalization)     |
| critic_obs_norm      | False    | False     | Consistent with actor                    |

---

## 7. Domain Randomization

Keep existing randomization from `SpotRoughEventCfg`, plus one addition:

| Event                    | Range / Config                     | Mode     |
|--------------------------|------------------------------------|----------|
| physics_material         | static=[0.5, 1.25], dynamic=[0.4, 1.0] | startup |
| add_base_mass            | [-5.0, +5.0] kg                    | startup  |
| reset_base               | pose: ±0.5m XY, ±π yaw            | reset    |
| reset_robot_joints       | pos: ±0.2 rad, vel: ±2.5 rad/s    | reset    |
| push_robot               | vel: ±0.5 m/s XY, every 10-15s    | interval |
| **base_external_force**  | **force: [-3.0, 3.0] N**          | **reset**|

The external force torque was previously zeroed out (`0.0, 0.0`). We enable it at ±3.0 N to
improve perturbation recovery — this directly targets the "falls when deployed" problem.

---

## 8. Environment Configuration

| Parameter            | Value  | Notes                              |
|----------------------|--------|------------------------------------|
| sim.dt               | 0.002  | 500 Hz physics                     |
| decimation           | 10     | 50 Hz control (matches deployment) |
| episode_length_s     | 20.0   | Standard                           |
| action_scale         | 0.25   | Keep — prevents overshooting       |
| action_clip          | [-1, 1]| Standard PPO output range          |
| obs_clip             | [-100, 100] | Prevents NaN propagation      |
| terrain              | ROUGH_TERRAINS_CFG | Stairs, slopes, boxes, etc. |
| terrain curriculum   | Enabled| Progressive difficulty             |
| max_init_terrain_lvl | 5      | Allow starting on medium terrain   |
| height_scanner       | 187-dim, 0.1m resolution, 1.6x1.0m (17x11 grid) | Terrain perception |
| contact_forces       | 4 feet | Gait detection                     |

---

## 9. Monitoring & Checkpointing

### TensorBoard

```bash
# On the H100 server (in a separate screen window)
tensorboard --logdir ~/IsaacLab/logs/rsl_rl/spot_rough --port 6006 --bind_all
```

Access from local machine: `http://172.24.254.24:6006`

### Key Metrics to Watch

| Metric                   | Healthy Range      | Red Flag                    |
|--------------------------|--------------------|-----------------------------|
| mean_reward              | Increasing trend   | Flatline after 5k iters     |
| mean_episode_length      | 15-20s (near max)  | Dropping below 5s           |
| policy_loss              | Decreasing         | Diverging / NaN             |
| value_loss               | Decreasing         | Diverging / NaN             |
| learning_rate            | 1e-4 to 3e-4       | Drops below 1e-5            |
| terrain_level (mean)     | Increasing 0→8+    | Stuck at level 0-1          |
| fps (steps/s)            | ~36,000            | Below 20,000                |

### Checkpoint Strategy

- `save_interval = 500` → checkpoint every ~4.6 minutes
- 60 checkpoints total across 30,000 iterations
- **Best model selection:** After training, evaluate top 5 checkpoints on all terrain levels
- Checkpoints saved to: `~/IsaacLab/logs/rsl_rl/spot_rough/<timestamp>/`

---

## 10. Deployment Considerations

After training, the policy must work in our standalone `spot_obstacle_course.py`. Key requirements:

1. **PhysX PD position control** - Kp=60, Kd=1.5 (Isaac Lab training defaults)
2. **Solver iterations** - 4 position / 0 velocity (GPU PhysX training config)
3. **Action scale** - 0.25 (must match training)
4. **Observation order** - [base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
   velocity_commands(3), joint_pos_rel(12), joint_vel_rel(12), last_action(12),
   height_scan(187)] = 235 total (GridPattern 17x11 = 187, not 160)
5. **CUDA tensors** - all ArticulationView setter calls must use `torch.Tensor` on `cuda:0`
6. **No observation normalization** - actor_obs_normalization=False

### Checkpoint Transfer

```bash
# From H100 to local machine
scp t2user@172.24.254.24:~/IsaacLab/logs/rsl_rl/spot_rough/<timestamp>/model_best.pt \
    C:/IsaacLab/logs/rsl_rl/spot_rough/48h_run/model_best.pt
```

---

## 11. File Structure

```
48h_training/
├── TRAINING_PLAN.md          ← This document
├── train_spot_rough_48h.sh   ← Main training launch script
├── spot_rough_48h_cfg.py     ← Custom env + PPO config overrides
└── eval_checkpoints.sh       ← Post-training evaluation script
```

---

## 12. Quick Start

```bash
# 1. SSH to server
ssh t2user@172.24.254.24

# 2. Start screen session
screen -S spot_48h

# 3. Activate environment
conda activate env_isaaclab
cd ~/IsaacLab

# 4. Upload training config
# (scp from local to server first)

# 5. Launch training
./isaaclab.sh -p /path/to/spot_rough_48h_cfg.py \
    --headless \
    --num_envs 8192

# 6. Detach: Ctrl+A, then D
# 7. Reconnect later: screen -r spot_48h
```

---

## 13. Risk Mitigations

| Risk                              | Mitigation                                    |
|-----------------------------------|-----------------------------------------------|
| Server crash mid-training         | Checkpoints every 500 iters (~5 min)          |
| Reward hacking / degenerate gait  | Gait reward (wt=10) + action smoothness (-2)  |
| Policy divergence                 | Adaptive LR + desired_kl=0.01 + grad clipping|
| Height scan all-1.0 bug           | Known issue — 20m Z-offset clips scan to 1.0; may need RayCaster offset fix on H100 |
| SSH disconnection                 | `screen` session survives disconnect          |
| GPU thermal throttling            | 8,192 envs keeps temp at ~49°C (well below throttle) |
| VRAM OOM                          | 8,192 envs uses ~10 GB of 96 GB available     |

---

**Next steps:** Write `spot_rough_48h_cfg.py` and `train_spot_rough_48h.sh`
