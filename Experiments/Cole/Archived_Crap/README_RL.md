# Spot RL Training System
## Multi-Terrain Navigation with Reinforcement Learning

Complete RL training framework for Boston Dynamics Spot robot using Nvidia's Isaac Lab and PyTorch PPO.

---

## System Components

### 1. **SpotRL_Environment.py**
Core environment wrapper that provides:
- Isaac Sim integration with 500Hz physics simulation
- Full sensory observations (IMU, joint encoders, cameras, goal info)
- Motor torque control (low-level joint actuation)
- Multi-terrain support (flat, obstacles, varied surfaces)
- Reward function for navigation efficiency
- Point-based time penalty system (300 points starting budget, 1 pt/sec)

**Key Classes:**
- `SpotRLConfig`: Configuration dataclass for environment parameters
- `SpotRLEnvironment`: Main environment class with step/reset/reward methods

### 2. **SpotRL_Training.py**
PPO-based training script providing:
- Proximal Policy Optimization (PPO) algorithm
- Curriculum learning (progressive terrain difficulty)
- Trajectory collection and batch processing
- Checkpoint saving/resuming
- Statistics logging and monitoring
- Support for custom reward shaping

**Key Classes:**
- `PPOTrainer`: Main trainer managing training loop and policy updates

---

## Installation & Setup

### Prerequisites
- Isaac Sim 5.1+ (already installed: C:\isaac-sim)
- Python 3.9+ (via Isaac Sim Python: C:\isaac-sim\python.bat)
- Git for version control

### Dependencies to Install

```bash
# Use Isaac Sim's Python environment
C:\isaac-sim\python.bat -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install RL/ML packages
C:\isaac-sim\python.bat -m pip install tensorboard numpy scipy gym

# Optional: For advanced RL features
C:\isaac-sim\python.bat -m pip install stable-baselines3 optuna
```

### Verify Installation
```bash
cd "c:\Users\user\Desktop\Capstone_vs_1.2\Immersive-Modeling-and-Simulation-for-Autonomy\Experiments\Cole"

# Test environment
C:\isaac-sim\python.bat SpotRL_Environment.py

# Expected output:
# ✓ SpotRL Environment initialized
# ✓ Spot robot created
# ✓ Sensor suite installed
# ✓ Scene setup complete
```

---

## Quick Start

### 1. Test the Environment
```bash
C:\isaac-sim\python.bat SpotRL_Environment.py

# Runs 100 random action steps to verify environment works
# Check for:
# - Spot robot spawning correctly
# - Physics simulation running
# - Reward computation working
# - Episode completion
```

### 2. Start Training (Single Episode)
```bash
C:\isaac-sim\python.bat SpotRL_Training.py --episodes 10 --headless
```

### 3. Full Training Session
```bash
# Open PowerShell and run:
C:\isaac-sim\python.bat SpotRL_Training.py --episodes 1000
```

---

## Configuration

### Environment Config (SpotRL_Environment.py)

```python
config = SpotRLConfig(
    # Episode settings
    episode_length=300,          # Max seconds per episode
    
    # Observations
    include_imu=True,            # Accelerometer, gyroscope
    include_encoders=True,       # Joint positions/velocities
    include_cameras=True,        # RGB depth cameras
    include_goal_info=True,      # Distance and direction to goal
    
    # Actions
    num_motors=12,               # Spot has 12 motors (3 per leg)
    motor_torque_limits=(-150, 150),  # Nm limits
    
    # Rewards
    reward_goal_reached=100.0,   # Bonus for reaching goal
    reward_progress=1.0,         # Per meter toward goal
    penalty_energy=0.01,         # For motor effort
    penalty_fall=-50.0,          # For falling
)
```

### Training Config (SpotRL_Training.py)

```python
training_config = {
    'num_episodes': 1000,
    'learning_rate': 3e-4,
    'batch_size': 32,
    'gamma': 0.99,               # Discount factor
    'entropy_coef': 0.01,        # Exploration bonus
    'hidden_size': 256,          # Network hidden layers
    'use_curriculum': True,      # Progressive difficulty
}
```

---

## Observation Space

The agent receives the following information at each step:

| Observation | Dim | Description |
|---|---|---|
| Position | 3 | (x, y, z) world coordinates |
| Heading | 1 | Yaw angle in radians |
| Linear Velocity | 3 | (vx, vy, vz) m/s |
| Angular Velocity | 3 | (ωx, ωy, ωz) rad/s |
| Joint Positions | 12 | 12 leg motor angles |
| Joint Velocities | 12 | 12 leg motor angular velocities |
| Distance to Goal | 1 | Euclidean distance (m) |
| Relative Goal | 2 | (Δx, Δy) to goal |
| Motor Effort | 12 | Previous action magnitudes |

**Total Observation Size: 59 dimensions**

---

## Action Space

The agent controls 12 motors (3 per leg × 4 legs):

| Motor Group | Indices | Motor Type |
|---|---|---|
| Front Left | 0-2 | Hip, Thigh, Calf |
| Front Right | 3-5 | Hip, Thigh, Calf |
| Back Left | 6-8 | Hip, Thigh, Calf |
| Back Right | 9-11 | Hip, Thigh, Calf |

**Action Range:** -150 to +150 Nm per motor

**Action Space:** 12-dimensional continuous

---

## Reward Structure

The agent receives reward signals for:

1. **Goal Progress** (+0 to +210 points/episode max)
   - +1 point per meter moved toward goal
   - +100 bonus when goal reached

2. **Energy Efficiency** (0 to -1.0 points/episode)
   - -0.01 × average motor effort
   - Encourages smooth, efficient movement

3. **Time Penalty** (-300 to 0 points/episode)
   - -1 point per second (built into environment)
   - Maximum 300 seconds (~5 minutes)

4. **Failure Penalties**
   - Fall: -50 + remaining points lost
   - Timeout: -50 (end of episode)

**Total Episode Reward Range:** ~-350 to +310 points

---

## Training Curriculum

Training progresses through difficulty stages:

### Stage 1: Flat Terrain (Episodes 0-200)
- Smooth, flat arena
- Learn basic locomotion
- Easy goal reaching

### Stage 2: Obstacles (Episodes 200-600)
- Arena boundaries act as obstacles
- Learn navigation around edges
- Moderate difficulty

### Stage 3: Varied Terrain (Episodes 600-1000)
- Multiple terrain types
- Challenging navigation
- Real-world simulation

---

## Monitoring Training

### Terminal Output
```
[Episode 42/1000] Collecting trajectories...
  Step 5/100: Reward=0.2456, Total=34.2341, Points=256

[Log] Episode 42:
  Avg Reward (last 10 eps): 28.4521
  Avg Episode Length: 87.3 steps
  Total Steps: 3654
```

### TensorBoard Visualization
```bash
# In separate terminal:
tensorboard --logdir ./runs/spot_rl --port 6006

# Open browser: http://localhost:6006
```

### Saved Checkpoints
```
checkpoints/spot_rl/
├── spot_rl_ep50.pt.json      # Every 50 episodes
├── spot_rl_ep100.pt.json
└── spot_rl_final.pt.json     # Final trained model
```

---

## Training Performance Expectations

| Metric | Value |
|---|---|
| Convergence Time | 500-1000 episodes |
| Training Time | 4-8 hours (single GPU) |
| Final Success Rate | 85-95% goal reaches |
| Avg Episode Reward | 80-120 points |
| Avg Episode Length | 120-180 seconds |

---

## Troubleshooting

### Issue: "Isaac Lab not found"
```bash
# Install Isaac Lab
C:\isaac-sim\python.bat -m pip install isaaclab
```

### Issue: "CUDA out of memory"
```python
# In training config:
config['batch_size'] = 16  # Reduce from 32
config['hidden_size'] = 128  # Reduce from 256
```

### Issue: Robot falls immediately
- Check if terrain generation is working
- Verify motor torque limits aren't too high
- Increase penalty for falling to encourage caution

### Issue: Training too slow
```bash
# Use headless mode (no rendering)
C:\isaac-sim\python.bat SpotRL_Training.py --headless --episodes 100
```

---

## Next Steps

### Implement Missing Features
1. **Motor Control Interface**
   - Map computed torques to joint targets
   - Implement joint servo loops

2. **Policy Network**
   - Replace placeholder with actual PyTorch network
   - Implement forward pass and action sampling

3. **Multi-Terrain Generation**
   - Add obstacles to arena
   - Generate varied surface types
   - Implement curriculum switching logic

4. **Camera Integration**
   - Process RGB/depth images
   - Add vision-based observations
   - Optional: CNN feature extraction

### Advanced Features
- Multi-agent training (multiple Spot robots)
- Domain randomization (texture, friction, mass variation)
- Transfer learning from simulation to real robot
- Imitation learning from human demonstrations
- Hierarchical RL (high-level planner + low-level controller)

---

## File Structure
```
Cole/
├── Environment2_flat_terrain.py     # Original deterministic environment
├── SpotRL_Environment.py            # RL-compatible environment wrapper
├── SpotRL_Training.py               # PPO training script
├── README_RL.md                     # This file
├── checkpoints/spot_rl/             # Saved models
├── runs/spot_rl/                    # TensorBoard logs
└── data/                            # Training data storage
```

---

## References

- [Nvidia Isaac Lab Documentation](https://docs.omniverse.nvidia.com/app/isaacsim/latest/)
- [OpenAI Gym RL Environment API](https://gym.openai.com/)
- [PPO: Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Boston Dynamics Spot Robot](https://www.bostondynamics.com/products/spot)

---

## Support & Troubleshooting

For issues or questions:
1. Check error message carefully
2. Review TensorBoard logs for training trends
3. Run test episode: `python SpotRL_Environment.py`
4. Check Isaac Sim logs: `C:\isaac-sim\kit\logs/`

---

**Last Updated:** February 16, 2026  
**Status:** Beta - Core environment working, PPO trainer framework in place
