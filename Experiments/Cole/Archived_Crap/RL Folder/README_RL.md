# Spot RL Training System for Obstacle-Aware Navigation

Complete reinforcement learning framework for training Boston Dynamics Spot quadruped to navigate cluttered environments with intelligent obstacle interaction.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Observation Space](#observation-space)
- [Action Space](#action-space)
- [Reward Function](#reward-function)
- [Training Configuration](#training-configuration)
- [Quick Start](#quick-start)
- [Files](#files)
- [Training Workflow](#training-workflow)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This RL system teaches Spot to:
1. **Navigate efficiently** using forward-dominant locomotion
2. **Interact intelligently** with obstacles (nudge light objects, bypass heavy ones)
3. **Maintain stability** during all maneuvers
4. **Reach waypoints** in cluttered circular arena

**Framework**: RSL-RL (PPO-based locomotion RL from ETH Zurich)  
**Simulation**: NVIDIA Isaac Sim with PhysX  
**Training**: Multi-environment parallel training on GPU

## 🏗️ Architecture

```
RL Folder/
├── spot_rl_env.py           # Core RL environment (Gym-style interface)
├── training_config.py       # Hyperparameters and training profiles
├── train_spot.py            # Training script (RSL-RL integration)
├── eval_policy.py           # Policy evaluation and visualization
├── README_RL.md             # This file
└── logs/                    # Training logs and checkpoints
    ├── checkpoints/
    ├── tensorboard/
    └── episodes/
```

## 👁️ Observation Space

**Total Dimension**: 92

| Component | Dim | Description |
|-----------|-----|-------------|
| **Proprioception** | 37 | |
| Joint positions | 12 | 4 legs × 3 joints (hip, knee, ankle) |
| Joint velocities | 12 | Angular velocities |
| Base orientation | 4 | Quaternion [w, x, y, z] |
| Base linear velocity | 3 | [vx, vy, vz] in body frame |
| Base angular velocity | 3 | [wx, wy, wz] |
| Base height | 1 | Distance from ground |
| Roll | 1 | Body roll angle |
| Pitch | 1 | Body pitch angle |
| **Navigation** | 3 | |
| Waypoint distance | 1 | Normalized distance to current waypoint |
| Waypoint heading | 1 | Heading error to waypoint |
| Progress | 1 | Fraction of waypoints completed |
| **Obstacle Sensing** | 35 | 5 nearest obstacles × 7 features |
| Per obstacle: | | |
| - Relative position | 2 | [rel_x, rel_y] |
| - Distance | 1 | Euclidean distance |
| - Mass (normalized) | 1 | Mass / Spot mass |
| - Friction | 1 | Surface friction coefficient |
| - Shape ID | 1 | Encoded shape type |
| - Is static | 1 | Binary: heavy/immovable |
| **Contact & Forces** | 5 | |
| Foot contacts | 4 | Binary contact for each foot |
| Collision magnitude | 1 | Total collision force |
| **Action History** | 3 | |
| Previous actions | 3 | [prev_vx, prev_vy, prev_omega] |

### Normalization

- **Distances**: Normalized by arena diameter (50m)
- **Angles**: Normalized to [-1, 1] range
- **Velocities**: Normalized by max velocity limits
- **Mass**: Normalized by Spot's mass (32.7 kg)

## 🎮 Action Space

**Dimension**: 3

| Action | Range | Description |
|--------|-------|-------------|
| Forward velocity (vx) | [-1.5, 1.5] m/s | Forward/backward locomotion |
| Lateral velocity (vy) | [-0.5, 0.5] m/s | Side-stepping (discouraged) |
| Angular velocity (ω) | [-1.0, 1.0] rad/s | Turning rate |

**Notes**:
- Actions are continuous
- Clipped to safe ranges
- Policy outputs scaled actions

## 🎁 Reward Function

### Components

| Component | Weight | Description |
|-----------|--------|-------------|
| **Progress Rewards** | | |
| Waypoint reached | +100.0 | Large bonus for reaching waypoint |
| Distance reduction | +2.0/m | Reward for moving toward waypoint |
| **Locomotion Quality** | | |
| Forward locomotion | +1.0 | Encourage natural forward gait |
| Lateral penalty | -0.5 | Penalize side-stepping |
| Backward penalty | -1.0 | Penalize backing up |
| **Stability** | | |
| Upright posture | +0.2 | Reward stable orientation |
| Height deviation | -1.0 | Penalize height changes |
| **Obstacle Interaction** | | |
| Successful nudge | +5.0 | Light obstacle pushed |
| Smart bypass | +3.0 | Efficiently avoiding obstacle |
| Failed nudge | -2.0 | Stuck pushing immovable object |
| Collision force | -0.5 | Per unit contact force |
| **Energy Efficiency** | | |
| Action smoothness | -0.1 | Penalize jerky movements |
| Energy penalty | -0.05 | Penalize high action magnitudes |
| **Terminal Conditions** | | |
| Fall | -100.0 | Fell over |
| Timeout | -10.0 | Ran out of time |
| **Time Penalty** | -0.01 | Per step (efficiency) |

### Total Reward Per Step

$$
r_t = r_{\text{progress}} + r_{\text{locomotion}} + r_{\text{stability}} + r_{\text{obstacle}} + r_{\text{energy}} + r_{\text{terminal}} + r_{\text{time}}
$$

## ⚙️ Training Configuration

### Default Profile

```python
from training_config import get_default_config

config = get_default_config()
# 4096 parallel environments
# 5000 training iterations
# PPO with adaptive learning rate
```

### Training Profiles

| Profile | Envs | Iterations | Use Case |
|---------|------|------------|----------|
| `debug` | 16 | 100 | Quick testing |
| `default` | 4096 | 5000 | Standard training |
| `fast` | 1024 | 5000 | Rapid prototyping |
| `stable` | 8192 | 5000 | Conservative, stable |
| `high_perf` | 16384 | 10000 | GPU clusters |

### Key Hyperparameters

```python
PPO Configuration:
  learning_rate: 3e-4
  clip_param: 0.2
  ppo_epoch: 8
  gamma: 0.99 (discount factor)
  lam: 0.95 (GAE lambda)
  
Environment:
  physics_dt: 1/500 sec (500 Hz physics)
  control_dt: 1/50 sec (50 Hz control)
  max_episode_length: 1000 steps
  
Network Architecture:
  Actor: [256, 256, 128]
  Critic: [256, 256, 128]
  Activation: ELU
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Isaac Sim must be installed
# RSL-RL framework
pip install rsl-rl

# Additional utilities
pip install torch tensorboard matplotlib
```

### 2. Run Training

```python
# From RL Folder/
python train_spot.py --config default --num_envs 4096
```

### 3. Monitor Training

```bash
# TensorBoard
tensorboard --logdir logs/tensorboard

# View in browser: http://localhost:6006
```

### 4. Evaluate Policy

```python
python eval_policy.py --checkpoint logs/checkpoints/model_5000.pt --episodes 10 --render
```

## 📁 Files

### Core Implementation

**`spot_rl_env.py`** (850 lines)
- `SpotRLEnv` class: Main RL environment
- `get_observations()`: Constructs observation vector
- `calculate_reward()`: Reward function
- `step()`: Execute environment step
- `ContactTracker`: Obstacle interaction logging

**`training_config.py`** (400 lines)
- `TrainingConfig`: Complete configuration dataclass
- `PPOConfig`: PPO algorithm parameters
- `EnvironmentConfig`: Simulation settings
- `RewardConfig`: Reward weights
- Pre-configured training profiles

### Training Scripts

**`train_spot.py`** (to be created)
- RSL-RL integration
- Multi-environment training loop
- Checkpoint saving
- TensorBoard logging

**`eval_policy.py`** (to be created)
- Policy evaluation
- Video recording
- Performance metrics

## 🔄 Training Workflow

### Phase 1: Baseline Locomotion (Iterations 0-1000)

Focus: Learn stable forward walking and basic waypoint navigation

```python
config = get_default_config()
config.rewards.waypoint_reached = 50.0  # Reduce initially
config.rewards.forward_locomotion = 2.0  # Emphasize
config.rewards.stability_reward = 0.5
```

**Expected**: Spot learns to walk forward, turn toward waypoints, maintain balance

### Phase 2: Obstacle Awareness (Iterations 1000-3000)

Focus: Learn to sense and respond to obstacles

```python
config.rewards.successful_nudge = 5.0
config.rewards.smart_bypass = 3.0
config.rewards.collision_penalty = 1.0  # Increase
```

**Expected**: Spot starts avoiding collisions, attempts pushing light objects

### Phase 3: Intelligent Interaction (Iterations 3000-5000)

Focus: Optimize nudge vs. bypass decisions

```python
config.rewards.failed_nudge_penalty = 3.0  # Increase
config.rewards.smart_bypass = 5.0  # Increase
```

**Expected**: Spot distinguishes pushable vs. immovable obstacles, chooses efficient strategies

### Curriculum Learning (Optional)

```python
# Start with fewer, lighter obstacles
if iteration < 1000:
    config.env.num_obstacles = 20
    # Increase light obstacle ratio
elif iteration < 3000:
    config.env.num_obstacles = 30
else:
    config.env.num_obstacles = 40  # Full difficulty
```

## 🔧 Hyperparameter Tuning

### Learning Rate Sweep

```python
from training_config import create_sweep_configs

configs = create_sweep_configs(
    get_default_config(),
    "ppo.learning_rate",
    [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
)

for config in configs:
    train(config)
```

### Reward Balancing

Monitor TensorBoard for:
- **Waypoint progress**: Should steadily increase
- **Collision rate**: Should decrease over time
- **Episode length**: Should increase (less falling)
- **Forward velocity**: Should be positive

Adjust rewards if:
- **Too many falls**: Increase `stability_reward`, decrease `forward_locomotion`
- **Not reaching waypoints**: Increase `waypoint_reached`, `distance_reduction`
- **Excessive collisions**: Increase `collision_penalty`
- **Inefficient paths**: Increase `smart_bypass`, decrease `failed_nudge_penalty`

## 🐛 Troubleshooting

### Training Instability

**Symptoms**: Reward drops suddenly, erratic behavior

**Solutions**:
```python
config.ppo.clip_param = 0.15  # More conservative
config.ppo.learning_rate = 1e-4  # Slower learning
config.ppo.max_grad_norm = 0.5  # Stricter gradient clipping
```

### Slow Convergence

**Symptoms**: Minimal improvement after 1000+ iterations

**Solutions**:
```python
config.ppo.learning_rate = 1e-3  # Faster learning
config.ppo.ppo_epoch = 4  # Fewer epochs per update
config.env.num_envs = 8192  # More parallel experience
```

### Memory Issues

**Symptoms**: CUDA out of memory errors

**Solutions**:
```python
config.env.num_envs = 1024  # Reduce parallel envs
config.ppo.num_mini_batches = 2  # Fewer mini-batches
# Or use smaller GPU, adjust accordingly
```

### Policy Exploits Reward

**Symptoms**: High reward but poor behavior (e.g., spinning instead of walking)

**Solutions**:
- Review reward function carefully
- Add constraints or penalties for unwanted behaviors
- Increase `action_smoothness`, `energy_penalty`
- Add explicit penalties for specific exploits

## 📊 Performance Metrics

### Key Metrics to Track

1. **Waypoints per Episode**: Target > 15/25
2. **Success Rate**: Episodes completing > 50%
3. **Collision Rate**: Collisions per episode < 10
4. **Nudge Success**: Light obstacles pushed > 70%
5. **Bypass Efficiency**: Path deviation < 2m
6. **Stability**: Fall rate < 5%
7. **Energy**: Average action magnitude < 0.7

### Evaluation Protocol

```python
# Run 100 episodes with fixed seed
python eval_policy.py --checkpoint best_model.pt \
                      --episodes 100 \
                      --seed 42 \
                      --save_stats
```

## 🎓 Advanced Topics

### Transfer Learning

Train on easier task first, then fine-tune:

```python
# Stage 1: No obstacles
config1 = get_default_config()
config1.env.num_obstacles = 0
train(config1, iterations=2000)

# Stage 2: With obstacles, load checkpoint
config2 = get_default_config()
train(config2, checkpoint="logs/model_2000.pt")
```

### Domain Randomization

Randomize physics parameters for robustness:

```python
config.env.mass_randomization = True  # ±20% robot mass
config.env.friction_randomization = True  # Vary terrain friction
config.env.force_randomization = True  # Add external forces
```

### Multi-Task Learning

Train single policy for multiple scenarios:

```python
# Mix arena sizes, obstacle densities, waypoint patterns
config.env.randomize_arena = True
config.env.randomize_waypoints = True
```

## 📚 References

- **RSL-RL**: https://github.com/leggedrobotics/rsl_rl
- **Isaac Sim**: https://docs.omniverse.nvidia.com/isaacsim/latest/
- **PPO Paper**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **Spot Specs**: Boston Dynamics Spot Technical Specifications

## 👤 Author

Cole  
MS for Autonomy Project  
February 2026

## 📄 License

Educational use for MS Capstone Project

---

**Next Steps**: See [train_spot.py](train_spot.py) for training script and [eval_policy.py](eval_policy.py) for evaluation examples.
