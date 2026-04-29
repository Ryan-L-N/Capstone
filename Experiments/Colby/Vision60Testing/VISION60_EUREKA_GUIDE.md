# Vision 60 Eureka Training Guide

## Complete How-To for Training Vision 60 Quadruped with LLM-Powered Rewards

---

## Table of Contents

1. [Overview](#1-overview)
2. [What is Eureka?](#2-what-is-eureka)
3. [Project Structure](#3-project-structure)
4. [Prerequisites](#4-prerequisites)
5. [Quick Start](#5-quick-start)
6. [Training Approaches](#6-training-approaches)
7. [Detailed Component Breakdown](#7-detailed-component-breakdown)
8. [Critical Lessons Learned](#8-critical-lessons-learned)
9. [Troubleshooting](#9-troubleshooting)
10. [Training Results & Current Status](#10-training-results--current-status)

---

## 1. Overview

### Goal
Train a **Ghost Robotics Vision 60** quadruped robot to achieve stable locomotion using **NVIDIA Eureka** - an LLM-based reward function generator.

### Target Capabilities
| Capability | Target Speed | Description |
|------------|--------------|-------------|
| Standing | 0 m/s | Stable upright stance for 10+ seconds |
| Walking | 0.3-0.5 m/s | Smooth forward locomotion |
| Trotting | 0.8-1.2 m/s | Diagonal leg coordination (FR+RL, FL+RR) |
| Running | 1.5-2.5 m/s | High-speed with flight phase |

### Robot Specifications
- **Joints**: 12 DoF (3 per leg × 4 legs)
  - Hip joints (0, 2, 4, 6): Sagittal rotation
  - Knee joints (1, 3, 5, 7): Sagittal rotation
  - Abduction joints (8, 9, 10, 11): Lateral rotation
- **Standing Height**: ~0.55-0.585m
- **Control Frequency**: 60 Hz physics, 30 Hz control
- **Test Arena**: 30ft × 30ft (9.14m × 9.14m)

---

## 2. What is Eureka?

### The Core Concept
**Eureka** is NVIDIA's system that uses GPT-4/LLMs to automatically design and iterate reward functions for reinforcement learning.

### How It Works
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Task Description│────▶│  GPT-4 (Eureka) │────▶│  Reward Function │
│  (Natural Language)│   │  (Code Generator)│    │  (Python Code)   │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                        ┌─────────────────────────────────────────┐
                        │        Isaac Sim Environment            │
                        │  1. Train policy with generated reward  │
                        │  2. Collect training metrics            │
                        │  3. Send feedback to LLM                │
                        │  4. LLM improves reward → iterate       │
                        └─────────────────────────────────────────┘
```

### Why Eureka?
- **No manual reward tuning** - LLM designs rewards from natural language
- **Iterative improvement** - Automatically refines based on training feedback
- **Human-level performance** - Published results show competitive reward design
- **Template fallback** - Works without API key using pre-defined templates

---

## 3. Project Structure

```
MS_for_autonomy/experimental_design_vision60_alpha/
├── README.md                    # Project overview
├── lessons_learned.md           # CRITICAL - Read before running!
│
├── code/
│   ├── eureka_integration/      # Eureka LLM-based reward generation
│   │   ├── eureka_reward_generator.py   # Core reward generator
│   │   ├── codex_integration.py         # OpenAI API wrapper
│   │   ├── vision60_eureka_env.py       # Isaac Sim environment
│   │   ├── train_with_eureka.py         # Main training script
│   │   └── README.md                    # Eureka-specific docs
│   │
│   ├── rl_training/             # Custom PPO training (non-Eureka)
│   │   ├── train_vision60_full.py       # Full training pipeline
│   │   └── vision60_env.py              # Environment wrapper
│   │
│   └── results/                 # Training outputs
│       ├── eureka_training/     # Eureka training runs
│       └── rl_training_full/    # Custom RL runs
│
└── phases/                      # Phase documentation
    ├── phase_1_urdf_validation.md
    ├── phase_2_standing_stability.md
    ├── phase_3_walking_baseline.md
    ├── phase_4_pretrained_policy.md
    └── phase_5_rl_training.md
```

---

## 4. Prerequisites

### Software Requirements
```bash
# Isaac Sim 5.1.0
# Python 3.11 (with Isaac Sim's conda environment)

# Activate the Isaac Sim environment
conda activate isaaclab311

# Required packages (most come with Isaac Sim)
pip install torch numpy
pip install openai  # For Eureka LLM mode
```

### API Keys (Optional - for full Eureka mode)
```bash
# Set OpenAI API key for GPT-4 reward generation
export OPENAI_API_KEY="sk-your-key-here"

# Optional: Organization ID
export OPENAI_ORG_ID="org-your-org-id"
```

### File Paths
Ensure the Vision60 URDF is accessible:
```
Vision 60/USMA Vison 60 Docs/vision60/urdf/vision60_v5.urdf
```

---

## 5. Quick Start

### Option A: Eureka Training with Templates (No API Key)
```bash
cd MS_for_autonomy/experimental_design_vision60_alpha/code/eureka_integration

# Run with pre-defined reward templates
python train_with_eureka.py --epochs 500
```

### Option B: Eureka Training with GPT-4 (Requires API Key)
```bash
export OPENAI_API_KEY="sk-..."

# Run with LLM-generated rewards
python train_with_eureka.py --use-llm --epochs 500
```

### Option C: Custom PPO Training (No Eureka)
```bash
cd MS_for_autonomy/experimental_design_vision60_alpha/code/rl_training

# Run custom PPO training
python train_vision60_full.py --epochs 250 --headless
```

### Option D: Train Specific Phase Only
```bash
# Train only the walking phase
python train_with_eureka.py --phase walking --epochs 200
```

---

## 6. Training Approaches

### Approach 1: Full Eureka Pipeline with LLM

**When to use**: You have an OpenAI API key and want automatic reward iteration.

**How it works**:
1. Eureka sends task description to GPT-4
2. GPT-4 generates Python reward function code
3. Training runs with generated reward
4. Metrics sent back to GPT-4
5. GPT-4 improves reward → iterate

**Code example**:
```python
from eureka_reward_generator import EurekaRewardGenerator

# Create generator with LLM enabled
generator = EurekaRewardGenerator(use_llm=True, model="gpt-4")

# Generate reward for walking phase
reward_config = generator.generate_reward_function(phase="walking")
print(reward_config["weights"])
print(reward_config["explanation"])
```

**API Cost**: ~$0.01-0.03 per reward generation (GPT-4)

---

### Approach 2: Eureka with Pre-defined Templates

**When to use**: No API key, or want predictable behavior.

**How it works**:
1. Uses pre-defined reward templates for each phase
2. No LLM calls - deterministic rewards
3. Still uses Eureka's multi-phase curriculum

**Available templates**:
| Phase | Key Rewards |
|-------|-------------|
| ROM Test | Joint movement, smooth transitions |
| Standing | Upright orientation, height tracking, stillness |
| Walking | Velocity tracking, gait regularity, stability |
| Trotting | Diagonal coordination, foot clearance |
| Running | Speed, stride length, flight phase |

**Code example**:
```python
from eureka_reward_generator import EurekaRewardGenerator

# Create generator WITHOUT LLM
generator = EurekaRewardGenerator(use_llm=False)

# Get template reward
reward_config = generator.generate_reward_function(phase="walking")
# Returns pre-defined weights and parameters
```

---

### Approach 3: Custom PPO Training

**When to use**: Want full control over reward function, or debugging.

**How it works**:
1. Hand-crafted reward function in `train_vision60_full.py`
2. Two-phase curriculum: Standing → Walking
3. Trot gait mimicry rewards
4. Goal-reaching rewards (3ft target)

**Key reward components**:
```python
# Standing phase rewards
W_STILLNESS = 0.5           # Zero velocity
W_STABLE_POSE = 0.5         # Default joint positions

# Walking phase rewards
W_FORWARD = 12.0            # Forward velocity tracking
W_BACKWARD_PENALTY = -25.0  # Penalize backward motion
W_GOAL_PROGRESS = 15.0      # Progress toward goal
W_GAIT_PHASE = 3.0          # Trot leg coordination
```

---

## 7. Detailed Component Breakdown

### 7.1 EurekaRewardGenerator (`eureka_reward_generator.py`)

**Purpose**: Generate reward functions using LLM or templates.

**Key methods**:
```python
# Initialize
generator = EurekaRewardGenerator(
    use_llm=True,           # Use GPT-4?
    model="gpt-4",          # Which model
    task_description="..."  # Custom task
)

# Generate reward
config = generator.generate_reward_function(phase="walking")
# Returns: {"name", "weights", "parameters", "description"}

# Update with training feedback
generator.update_from_training({
    "mean_reward": 125.5,
    "success_rate": 0.72,
})

# Save/load configuration
generator.save_reward_config("rewards.json")
generator.load_reward_config("rewards.json")
```

**Template structure**:
```python
{
    "name": "Walking Gait",
    "description": "Smooth walking at 0.3-0.5 m/s",
    "weights": {
        "forward_velocity": 10.0,
        "lateral_velocity_penalty": -2.0,
        "upright_orientation": 3.0,
        "action_rate_penalty": -0.1,
        # ... more components
    },
    "parameters": {
        "target_velocity": [0.4, 0.0, 0.0],
        "gait_frequency": 1.5,  # Hz
    }
}
```

---

### 7.2 CodexClient (`codex_integration.py`)

**Purpose**: Interface with OpenAI API for code generation.

**Supported models**:
| Model | Context | Cost/1K | Best For |
|-------|---------|---------|----------|
| gpt-4 | 8K | $0.03 | Highest quality |
| gpt-4-turbo | 128K | $0.01 | Long context |
| gpt-4o | 128K | $0.005 | Fast + cheap |
| o1-mini | 128K | $0.003 | Advanced reasoning |
| gpt-3.5-turbo | 16K | $0.0005 | Budget |

**Usage**:
```python
from codex_integration import CodexClient, CodexConfig, setup_codex

# Quick setup
codex = setup_codex(model="gpt-4")

# Or with config
config = CodexConfig(
    api_key="sk-...",
    model="gpt-4-turbo",
    temperature=0.2,
    max_tokens=2048,
)
codex = CodexClient(config)

# Generate reward code
task_spec = {"robot": "Vision60", "arena_size": {"m": 9.14}}
result = codex.generate_reward_code(task_spec, phase="walking")

print(result["reward_code"])     # Python code
print(result["weights"])          # Weight dict
print(result["components"])       # Reward components
print(result["code_valid"])       # Syntax check result

# Iterate on reward
improved = codex.iterate_reward(
    current_reward=result,
    training_metrics={"success_rate": 0.5},
    feedback="Robot falls during turns"
)

# Check API usage
stats = codex.get_usage_stats()
print(f"Tokens: {stats['total_tokens']}, Cost: ${stats['estimated_cost']:.4f}")
```

---

### 7.3 Vision60EurekaEnv (`vision60_eureka_env.py`)

**Purpose**: Isaac Sim environment with multi-phase curriculum.

**Key features**:
- Folded initialization → Standing → Walking → Trotting → Running
- 30ft × 30ft test arena
- Eureka reward function integration
- Joint limit enforcement

**Phase configuration**:
```python
# Folded position (robot low, compressed)
FOLDED_POSITION = np.array([
    0.2, 0.2, 0.2, 0.2,      # Hips
    2.5, 2.5, 2.5, 2.5,      # Knees (bent)
    0.0, 0.0, 0.0, 0.0,      # Abduction
])

# Standing position
STANDING_POSITION = np.array([
    0.9, 0.9, 0.9, 0.9,      # Hips
    1.67, 1.67, 1.67, 1.67,  # Knees
    0.03, 0.03, -0.03, -0.03, # Abduction
])

# Gait velocities
GAIT_VELOCITIES = {
    "walking": 0.4,   # m/s
    "trotting": 1.0,
    "running": 2.0,
}
```

**Usage**:
```python
from vision60_eureka_env import Vision60EurekaEnv

# Create environment
env = Vision60EurekaEnv(headless=True, device="cuda")

# Set training phase
env.set_phase("walking")

# Reset and step
obs = env.reset()
for step in range(1000):
    action = policy(obs)  # Your policy
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()
```

---

### 7.4 Training Script (`train_with_eureka.py`)

**Purpose**: PPO training loop with Eureka integration.

**Curriculum phases**:
1. **ROM Test** (50 epochs): Test all joints through range
2. **Standing** (50 epochs): Stable upright posture
3. **Walking** (100 epochs): 0.4 m/s forward
4. **Trotting** (150 epochs): 1.0 m/s diagonal coordination
5. **Running** (150 epochs): 2.0 m/s with flight phase

**Command line options**:
```bash
python train_with_eureka.py \
    --use-llm          # Enable GPT-4 reward generation
    --epochs 500       # Total epochs (overrides per-phase)
    --phase walking    # Train specific phase only
    --headless         # No visualization
    --device cuda      # GPU training
```

**Output directory**: `code/results/eureka_training/<timestamp>/`

---

### 7.5 Custom PPO Training (`train_vision60_full.py`)

**Purpose**: Full-featured training without Eureka dependency.

**Key configuration**:
```python
# Simulation
SIM_DT = 1/60               # 60 Hz physics
CONTROL_DECIMATION = 2       # 30 Hz control
MAX_EPISODE_STEPS = 500

# Robot control (50% BOOSTED for locomotion)
KP = 200.0                   # PD stiffness
KD = 5.0                     # PD damping
ACTION_SCALE = 0.375         # Action scaling

# Training
EPOCHS = 250
STEPS_PER_EPOCH = 512
MINIBATCH_SIZE = 128
LEARNING_RATE = 3e-4
```

**Curriculum**:
- Standing phase: First 10 steps of each episode
- Walking phase: After standing, velocity command ramps up

**Reward structure**:
| Component | Phase | Weight | Purpose |
|-----------|-------|--------|---------|
| forward | Walking | 12.0 | Track velocity |
| backward_penalty | Walking | -25.0 | No reversing |
| goal_progress | Walking | 15.0 | Reach 3ft target |
| orientation | Both | 2.0 | Stay upright |
| gait_phase | Walking | 3.0 | Trot coordination |
| stillness | Standing | 0.5 | Zero velocity |

---

## 8. Critical Lessons Learned

### 8.1 Gravity Sign Convention
```python
# WRONG - Robot flies upward!
physics_context.set_gravity(9.81)

# CORRECT - Robot falls with gravity
physics_context.set_gravity(-9.81)
```

### 8.2 Joint Angle Direction (Counter-Intuitive!)
```python
# SMALLER angles = LOWER robot position
FOLDED = {"hip": 0.6, "knee": 1.2}  # h ≈ 0.41m

# LARGER angles = HIGHER robot position
STANDING = {"hip": 0.9, "knee": 1.8}  # h ≈ 0.55m
```

### 8.3 Joint Order Matters!
USMA policies use **grouped order**, not sequential:
```python
# Policy order (USMA):
# [hip_FR, hip_FL, hip_RR, hip_RL, knee_FR, knee_FL, ...]

# Isaac Sim order (URDF):
# [joint_0, joint_1, joint_2, ...]  (may differ!)

# Always use mapping functions!
def to_isaac_order(policy_data):
    result = np.zeros(12)
    for p in range(12):
        result[policy_to_isaac_idx[p]] = policy_data[p]
    return result
```

### 8.4 Configuration Discrepancies
Two config files have different values:
| Parameter | Vision60.yaml | vision60_config.py |
|-----------|--------------|-------------------|
| Kp | 85 | 80 |
| Kd | 2.0 | 0.5 |
| action_scale | 0.5 | 0.25 |

**Recommendation**: Use Vision60.yaml values (more stable).

### 8.5 Initialization Protocol
```python
# Working sequence:
1. Start at folded position (h ≈ 0.41m)
2. Smooth interpolation over 30 steps
3. Hold at standing position
4. Then enable RL control

# Smooth interpolation function
def smooth_step(t):
    return t * t * (3 - 2 * t)  # S-curve
```

---

## 9. Troubleshooting

### Robot Flies Upward
**Cause**: Positive gravity value
**Fix**: Use `physics_context.set_gravity(-9.81)`

### Robot Collapses Immediately
**Causes**:
1. Wrong joint angles (larger ≠ lower!)
2. Missing initialization sequence
3. PD gains too low

**Fix**: Use proven standing position and init protocol.

### Robot Walks Backward
**Cause**: Joint order mismatch
**Fix**: Verify policy-to-Isaac joint mapping.

### Training Doesn't Converge
**Causes**:
1. Reward imbalance (too much forward reward before stable)
2. Termination too strict
3. Missing curriculum

**Fix**: Use curriculum learning - stand before walk.

### API Errors (Eureka LLM mode)
**Causes**:
1. Missing API key
2. Rate limiting
3. Model not available

**Fix**:
```bash
export OPENAI_API_KEY="sk-..."
# Or use template mode (no API needed)
python train_with_eureka.py  # No --use-llm flag
```

### Isaac Sim Crashes
**Causes**:
1. Memory leak during resets
2. Invalid physics state

**Fix**:
```python
# Periodic garbage collection
if reset_count % 5 == 0:
    gc.collect()
    torch.cuda.empty_cache()
```

---

## 10. Training Results & Current Status

### Phase Status (as of Feb 4, 2026)

| Phase | Status | Notes |
|-------|--------|-------|
| URDF Validation | ✅ Complete | 12 DOF confirmed |
| Standing Stability | ✅ Complete | 60+ second stance |
| Walking Baseline | ✅ Complete | ~60 step episodes |
| Pre-trained Policy | ⏸️ Paused | Sim mismatch issues |
| RL Training | 🔄 Active | 70% complete |

### Latest Eureka Training (Feb 2, 2026)

| Phase | Epochs | Final Reward | Success Rate |
|-------|--------|--------------|--------------|
| ROM Test | 50 | 0.74 | 8% |
| Standing | 1 | 7.4 | 100% |
| Walking | 1 | 4.3 | 100% |
| Trotting | 150 | Variable | Unstable |
| Running | 150 | Negative | Failed |

**Key Findings**:
- ROM and Standing converge well
- Trotting/Running need more tuning
- Reward variance high during faster gaits

### Best Custom PPO Results

- **Episode length**: 251 steps (near max 256)
- **Standing**: Stable for full episodes
- **Walking**: Forward locomotion achieved
- **Checkpoints**: 32 saved in `rl_training_full/20260131_082209/`

---

## Commands Reference

```bash
# Eureka with templates (recommended start)
python train_with_eureka.py --epochs 500

# Eureka with GPT-4
export OPENAI_API_KEY="sk-..."
python train_with_eureka.py --use-llm --epochs 500

# Custom PPO training
python train_vision60_full.py --epochs 250 --headless

# Train specific phase
python train_with_eureka.py --phase walking --epochs 200

# Continue from checkpoint
python train_vision60_full.py --load path/to/policy.pt --epochs 100

# ROM test only
python rom_test.py --visualize
```

---

## File Quick Reference

| Task | File | Location |
|------|------|----------|
| Eureka training | `train_with_eureka.py` | `code/eureka_integration/` |
| Custom PPO | `train_vision60_full.py` | `code/rl_training/` |
| Reward generator | `eureka_reward_generator.py` | `code/eureka_integration/` |
| OpenAI integration | `codex_integration.py` | `code/eureka_integration/` |
| Environment | `vision60_eureka_env.py` | `code/eureka_integration/` |
| Lessons learned | `lessons_learned.md` | Root |
| Project overview | `README.md` | Root |

---

**Author**: AI2C Capstone Team
**Last Updated**: February 4, 2026
