# 48-Hour Optimized Training Configuration
## ANYmal-C Rough Terrain - Locomotion / Proprioception / Joint Efficiency

**Date:** February 2025  
**Platform:** NVIDIA H100 NVL (96GB VRAM)  
**Task:** Isaac-Velocity-Rough-Anymal-C-v0  
**Estimated Duration:** ~48 hours  
**Total Timesteps:** ~3.3 billion  

---

## Table of Contents
1. [Training Objectives](#training-objectives)
2. [Environment Configuration](#environment-configuration)
3. [PPO Hyperparameter Optimization](#ppo-hyperparameter-optimization)
4. [Reward Engineering](#reward-engineering)
5. [Iteration Calculation](#iteration-calculation)
6. [Deployment Instructions](#deployment-instructions)
7. [Monitoring](#monitoring)
8. [Recovery Procedures](#recovery-procedures)
9. [Expected Outcomes](#expected-outcomes)

---

## Training Objectives

This 48-hour training run targets three specific capabilities:

### 1. Locomotion Quality
The policy should learn robust, high-quality walking gaits that reliably track commanded
velocities across rough terrain. Enhanced velocity tracking rewards (1.5x default) push the
policy to develop more precise locomotion behaviors over the extended training period.

### 2. Proprioceptive Awareness
The robot must leverage its internal joint state information (positions, velocities, torques)
to maintain stability. By enabling flat orientation rewards and increasing base stability
penalties, the policy is forced to develop strong proprioceptive control - reading its own
body state to maintain balance rather than relying solely on external cues.

### 3. Joint Efficiency
The actuators should use minimal energy and produce smooth, efficient motions. The default
configuration has negligible torque penalties (-1e-5), allowing the policy to use brute-force
torques. Our configuration increases torque penalties 20x and acceleration penalties 10x,
forcing the policy to discover efficient actuation strategies.

---

## Environment Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Environments** | 8,192 | Safe sustained thermal (65% GPU, 49°C, 188W from stress test) |
| **Simulation dt** | 0.005s | Default physics timestep |
| **Decimation** | 4 | Policy runs at 50 Hz |
| **Episode length** | 20s | Standard episode duration |
| **Terrain** | Rough (curriculum) | Pyramid stairs, random rough, boxes, slopes |
| **Robot** | ANYmal-C | 12-DOF quadruped with LSTM actuator network |
| **Actuator** | ActuatorNetLSTM | anydrive_3_lstm_jit.pt, effort_limit=80 Nm |

### Observations (Policy Input)
| Observation | Dimension | Noise | Purpose |
|-------------|-----------|-------|---------|
| base_lin_vel | 3 | ±0.1 | Body linear velocity (proprioception) |
| base_ang_vel | 3 | ±0.2 | Body angular velocity (proprioception) |
| projected_gravity | 3 | ±0.05 | Gravity direction in body frame |
| velocity_commands | 3 | None | Target velocity commands |
| joint_pos (relative) | 12 | ±0.01 | Joint positions (proprioception) |
| joint_vel (relative) | 12 | ±1.5 | Joint velocities (proprioception) |
| last_action | 12 | None | Previous action (action history) |
| height_scan | 187 | ±0.1 | Terrain heightmap |
| **Total** | **235** | | |

> **Proprioception note:** 33 of 235 observation dimensions (14%) are direct proprioceptive
> signals (joint_pos, joint_vel, base_lin_vel, base_ang_vel, gravity). The reward engineering
> below forces the policy to pay attention to these signals.

### Actions
| Action | Dimension | Scale | Type |
|--------|-----------|-------|------|
| joint_pos | 12 | 0.5 | Position targets for all joints |

---

## PPO Hyperparameter Optimization

| Parameter | Default | Optimized | Rationale |
|-----------|---------|-----------|-----------|
| **Actor hidden dims** | [512, 256, 128] | **[1024, 512, 256]** | Deeper network for complex proprioceptive reasoning; H100 has plenty of VRAM |
| **Critic hidden dims** | [512, 256, 128] | **[1024, 512, 256]** | Match actor capacity for accurate value estimation |
| **Learning rate** | 1e-3 | **3e-4** | Lower LR prevents oscillation in long training; adaptive schedule self-regulates |
| **GAE lambda** | 0.95 | **0.97** | Better long-horizon credit assignment for locomotion episodes |
| **Num learning epochs** | 5 | **8** | More updates per batch = more sample efficient |
| **Num mini-batches** | 4 | **8** | Larger effective batch, smoother gradients on H100 |
| **Desired KL** | 0.01 | **0.008** | Tighter policy constraint prevents destructive updates |
| **Init noise std** | 1.0 | **0.8** | Less random initial exploration = faster initial convergence |
| **Save interval** | 50 | **500** | Reduce disk I/O for 17,000 iteration run (34 checkpoints vs 340) |
| **Activation** | ELU | ELU | No change - works well for locomotion |
| **Gamma** | 0.99 | 0.99 | No change - standard discount factor |
| **Clip param** | 0.2 | 0.2 | No change - standard PPO clipping |
| **Entropy coeff** | 0.005 | 0.005 | No change - sufficient exploration |
| **Max grad norm** | 1.0 | 1.0 | No change - gradient clipping |

### Network Architecture
```
Actor:  obs(235) -> Linear(1024) -> ELU -> Linear(512) -> ELU -> Linear(256) -> ELU -> Linear(12)
Critic: obs(235) -> Linear(1024) -> ELU -> Linear(512) -> ELU -> Linear(256) -> ELU -> Linear(1)
```
Total parameters: ~1.5M (vs ~500K with default [512, 256, 128])

### Mini-batch configuration
- Steps per env: 24
- Batch size: 8,192 × 24 = 196,608 transitions/iteration
- Mini-batch size: 196,608 / 8 = 24,576 transitions/mini-batch
- Total gradient steps per iteration: 8 epochs × 8 mini-batches = 64

---

## Reward Engineering

### Modification Summary

| Reward Term | Default Weight | Optimized Weight | Multiplier | Category |
|-------------|---------------|-----------------|------------|----------|
| track_lin_vel_xy_exp | 1.0 | **1.5** | 1.5x | Locomotion |
| track_ang_vel_z_exp | 0.5 | **0.75** | 1.5x | Locomotion |
| lin_vel_z_l2 | -2.0 | -2.0 | 1x | Stability |
| ang_vel_xy_l2 | -0.05 | **-0.1** | 2x | Proprioception |
| flat_orientation_l2 | 0.0 | **-1.0** | ENABLED | Proprioception |
| dof_torques_l2 | -1e-5 | **-0.0002** | 20x | Efficiency |
| dof_acc_l2 | -2.5e-7 | **-2.5e-6** | 10x | Efficiency |
| action_rate_l2 | -0.01 | **-0.025** | 2.5x | Smoothness |
| dof_pos_limits | 0.0 | **-5.0** | ENABLED | Safety |
| feet_air_time | 0.125 | **0.25** | 2x | Gait Quality |
| undesired_contacts | -1.0 | -1.0 | 1x | Safety |

### Reward Balance Analysis

**Positive rewards (max possible per step):**
- track_lin_vel_xy_exp: up to 1.5 (when perfectly tracking)
- track_ang_vel_z_exp: up to 0.75 (when perfectly tracking)  
- feet_air_time: up to ~0.25 (proper gait timing)
- **Max positive: ~2.5**

**Negative penalties (typical magnitudes per step):**
- lin_vel_z_l2: ~-0.1 to -0.5 (depends on bouncing)
- ang_vel_xy_l2: ~-0.01 to -0.1 (depends on tilting)
- flat_orientation_l2: ~-0.1 to -0.5 (depends on body roll/pitch)
- dof_torques_l2: ~-0.5 to -2.0 (depends on torque usage)
- dof_acc_l2: ~-0.01 to -0.1
- action_rate_l2: ~-0.01 to -0.1
- undesired_contacts: ~0 to -1.0 (binary events)
- dof_pos_limits: ~0 to -5.0 (only when near limits)

> The reward balance ensures the policy CAN achieve positive rewards through good locomotion,
> but MUST be efficient and stable to maximize total return. The heavy torque/acceleration
> penalties are the primary drivers of joint efficiency learning.

### Why This Works for Proprioception

1. **flat_orientation_l2 = -1.0**: The robot must maintain an upright posture. The ONLY way
   to do this is by reading its proprioceptive state (projected_gravity, base_ang_vel) and
   reacting to maintain balance.

2. **ang_vel_xy_l2 = -0.1**: Penalizes excessive roll/pitch angular velocity. Forces the
   policy to develop stabilization reflexes based on proprioceptive feedback.

3. **dof_torques_l2 = -0.0002**: With heavy torque penalties, the policy cannot brute-force
   its way through terrain. It must learn precise, efficient joint control using proprioceptive
   information about current joint states and velocities.

4. **dof_pos_limits = -5.0**: Respecting joint limits requires awareness of current joint
   positions - a fundamental proprioceptive capability.

---

## Iteration Calculation

```
Environment count:     8,192
Steps per env/iter:    24
Steps per iteration:   8,192 × 24 = 196,608
Throughput (stress):   ~36,000 steps/s at 8K envs
Collection time:       196,608 / 36,000 ≈ 5.5s
Learning time (est):   ~4-5s (8 epochs × 8 mini-batches with [1024,512,256])
Total per iteration:   ~10s estimated

48 hours = 172,800 seconds
Iterations in 48h:     172,800 / 10 ≈ 17,280
Set max_iterations:    17,000 (with safety buffer)

Total timesteps:       17,000 × 196,608 = 3,342,336,000 (3.3 billion)
Checkpoints (500 int): 34 checkpoints × ~175MB ≈ 6GB storage
```

---

## Deployment Instructions

### Step 1: Upload Files to H100
```bash
# From local machine (via h100_run.py or SCP)
cd "Capstone/Experiments/Alex/48h_training"
scp train_48h_anymal_c.py t2user@172.24.254.24:~/
scp launch_48h.sh t2user@172.24.254.24:~/
scp monitor_48h.sh t2user@172.24.254.24:~/
```

### Step 2: SSH and Start Screen
```bash
ssh t2user@172.24.254.24
# Password: !QAZ@WSX3edc4rfv

# Create a screen session
screen -S train48h
```

### Step 3: Launch Training
```bash
bash ~/launch_48h.sh
```

### Step 4: Detach Screen
```
Press: Ctrl+A, then D
```
This detaches the screen session. Training continues in the background.

### Step 5: Disconnect SSH
```bash
exit
```
CRITICAL: Only ONE SSH session at a time. Disconnect after launching.

### Checking Progress (from any SSH session)
```bash
ssh t2user@172.24.254.24
bash ~/monitor_48h.sh
exit  # Disconnect immediately after checking
```

---

## Monitoring

### Quick Status Check
```bash
bash ~/monitor_48h.sh
```
Shows: GPU status, process status, latest checkpoint, progress estimate, system health.

### View Training Output
```bash
# Reattach to screen session
screen -r train48h
# Detach again: Ctrl+A, D
```

### View TensorBoard Logs (from local machine)
```bash
# Download logs
scp -r t2user@172.24.254.24:~/IsaacLab/logs/rsl_rl/anymal_c_48h/ ./48h_logs/
# View locally
tensorboard --logdir ./48h_logs/
```

### Key Metrics to Watch
1. **Mean reward**: Should steadily increase. Expect slower start due to heavier penalties.
2. **Mean episode length**: Should increase as robot learns to walk without falling.
3. **Terrain level**: Should increase with curriculum. Target: level 5+ by end of training.
4. **GPU temperature**: Should stay at ~49°C with 8K envs. Alert if >70°C.
5. **Learning rate**: Adaptive - should decrease over time as policy stabilizes.

---

## Recovery Procedures

### If Training Crashes
```bash
ssh t2user@172.24.254.24
screen -S train48h
bash ~/launch_48h.sh --resume
# Ctrl+A, D to detach
exit
```
The `--resume` flag automatically finds the latest checkpoint and continues.

### If Server Crashes
1. Wait for physical recovery (as documented in previous incidents)
2. SSH in after recovery
3. Resume from last checkpoint:
```bash
bash ~/launch_48h.sh --resume
```

### If You Need to Resume from Specific Checkpoint
```bash
bash ~/launch_48h.sh --resume --checkpoint /home/t2user/IsaacLab/logs/rsl_rl/anymal_c_48h/<run_name>/model_XXXX.pt
```

### If GPU Temperature Gets Too High
The H100 NVL has thermal throttling at ~83°C. At 8K envs we measured 49°C sustained, so
this should not be an issue. If it occurs:
```bash
# Reduce environment count on resume
# Edit launch_48h.sh: change --num_envs 8192 to --num_envs 4096
bash ~/launch_48h.sh --resume --num_envs 4096
```

---

## Expected Outcomes

### Compared to 1-Hour Training Baseline
| Metric | 1-Hour (1400 iter) | 48-Hour Expected |
|--------|--------------------|------------------|
| Reward | 10.76 | 15-25+ |
| Terrain level | 4.71 | 7-9 |
| Gait quality | Basic walking | Efficient, smooth gaits |
| Joint efficiency | Unoptimized | Energy-efficient actuation |
| Proprioception | Passive | Active stabilization |

### What to Expect at Different Stages
- **0-2 hours (~1,200 iter):** Initial learning, reward may be lower than default due to
  heavier penalties. Policy discovers basic locomotion.
- **2-8 hours (~5,000 iter):** Rapid improvement. Policy learns to walk while minimizing
  torques. Gait patterns emerge.
- **8-24 hours (~12,000 iter):** Refinement phase. Policy optimizes joint efficiency,
  develops robust proprioceptive reflexes. Terrain curriculum advances.
- **24-48 hours (~17,000 iter):** Fine-tuning. Marginal improvements in efficiency and
  robustness. Policy reaches near-optimal for this reward configuration.

### Files Produced
```
~/IsaacLab/logs/rsl_rl/anymal_c_48h/<run_name>/
├── model_500.pt          # Checkpoint at iteration 500
├── model_1000.pt         # Checkpoint at iteration 1000
├── ...                   # Every 500 iterations
├── model_17000.pt        # Final model
├── reward_config.txt     # Reward weights used
├── agent_config.txt      # PPO config used
└── events.out.tfevents.* # TensorBoard logs
```
