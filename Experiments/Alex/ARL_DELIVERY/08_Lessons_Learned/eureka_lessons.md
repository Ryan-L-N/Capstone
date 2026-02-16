# Lessons Learned – NVIDIA Eureka Integration

## Overview

This document tracks lessons learned, pitfalls avoided, and best practices discovered during the integration of NVIDIA Eureka for automated reward function design in grass terrain navigation.

> 🎯 **Goal**: Use Eureka's LLM-powered reward evolution to outperform the SpotFlatTerrainPolicy baseline on grass terrain.

---

## Document Structure

1. **Environment Setup** – Getting Eureka running with Isaac Sim/IsaacGym
2. **Do Not Repeat (DNR)** – Mistakes to avoid
3. **Best Practices** – Successful approaches
4. **Reward Engineering Insights** – What Eureka discovered
5. **Performance Analysis** – Training observations
6. **Phase-by-Phase Notes** – Chronological learnings

---

## Environment Setup

### EUR-001: ✅ Eureka Port to IsaacLab Complete

**Issue**: Original Eureka requires Python 3.8 and IsaacGym Preview 4, incompatible with Isaac Sim 5.1.0.

**Root Cause**: Eureka was built for IsaacGym's VecTask API, not IsaacLab's DirectRLEnv.

**Solution**: Created `eureka_isaaclab/` - a complete port of Eureka to IsaacLab:

```
eureka_isaaclab/
├── eureka_isaaclab.py      # Main evolution loop
├── reward_generator.py     # LLM prompts & validation
├── training_runner.py      # IsaacLab training integration
├── run_eureka_grass.py     # Entry point
└── envs/
    ├── spot_grass_env.py   # Grass navigation environment
    └── spot_grass_env_cfg.py
```

**Key Adaptations**:
1. Changed from `VecTask` to `DirectRLEnv` base class
2. Changed reward pattern from `compute_reward()` to `_get_rewards()`
3. Adapted to RSL-RL instead of rl_games
4. Works with Python 3.11 (Isaac Sim 5.1.0 requirement)

**Status**: ✅ Implemented (February 4, 2026)

---

### EUR-002: ✅ IsaacGym vs Isaac Sim Environment Compatibility

**Issue**: Eureka was built for IsaacGym (Preview 4), not Isaac Sim 5.1.0

**Root Cause**: Different APIs between IsaacGym and IsaacLab:
- IsaacGym: `VecTask` with `compute_reward(self, actions)`
- IsaacLab: `DirectRLEnv` with `_get_rewards(self)`

**Solution**: Created abstraction that:
1. Injects `compute_eureka_reward()` method into environment
2. Overrides `_get_rewards()` to call Eureka reward
3. Maintains compatibility with RSL-RL training

```python
# Eureka reward injection pattern
def _get_rewards(self) -> torch.Tensor:
    total_reward, self.reward_components = self.compute_eureka_reward()
    return total_reward
```

**Key Points**:
- Environment code is dynamically modified with new reward function
- Reward components are logged to `self.extras` for tensorboard
- Maintains full IsaacLab compatibility

**Status**: ✅ Implemented (February 4, 2026)

---

### EUR-003: ✅ Claude API Integration Complete

**Issue**: OpenAI API quota exceeded, needed alternative LLM provider.

**Solution**: Added Claude (Anthropic) support as primary LLM provider:

```powershell
# Claude (recommended)
$env:ANTHROPIC_API_KEY = "your-anthropic-key"

# Or OpenAI
$env:OPENAI_API_KEY = "your-openai-key"
```

**Key Points**:
- Claude Sonnet 4 is the default model (excellent code generation)
- Both Claude and GPT-4o supported via `--model` flag
- Auto-detects provider from model name prefix (`claude-*` vs `gpt-*`)

**Supported Models**:
- Claude: `claude-sonnet-4-20250514`, `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`
- OpenAI: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`

**Status**: ✅ Implemented (February 4, 2026)

---

### EUR-004: ✅ Training Integration Architecture

**Issue**: Need to connect Eureka reward evolution to actual RL training.

**Solution**: Created multi-layer training integration:

```
eureka_isaaclab/
├── train_eureka_env.py       # Subprocess training script
├── isaaclab_training.py      # Direct PPO training loop
├── isaaclab_env_registry.py  # Environment registration
└── training_runner.py        # Training orchestration
```

**Training Approaches**:
1. **Direct PPO** (`isaaclab_training.py`):
   - Self-contained PPO implementation
   - SimpleActorCritic neural network
   - No external dependencies beyond PyTorch

2. **RSL-RL Integration** (`train_eureka_env.py`):
   - Attempts RSL-RL first (if available)
   - Falls back to direct PPO
   - Subprocess isolation for stability

3. **IsaacLab Workflow**:
   - Uses `isaaclab.scripts.train` command
   - Native integration with IsaacLab ecosystem

**Key Architecture**:
```python
# Training is called per reward candidate
result = self._run_training(env_file, iteration, idx)
# Returns: {'success_rate': 0.43, 'avg_reward': 156.2, 'error': False}
```

**Status**: ✅ Implemented (February 4, 2026)

---

### EUR-005: [PENDING] OpenAI API Configuration

---

## Do Not Repeat (DNR)

### DNR-EUR-001: [PENDING] TBD

**What Went Wrong**: TBD

**Consequence**: TBD

**Prevention**: TBD

**Status**: 🔄 Waiting for first Eureka run

---

## Best Practices

### BP-EUR-001: [PENDING] Effective Task Descriptions for Eureka

**Context**: The task description provided to Eureka influences reward generation quality.

**Best Practice Template**:
```yaml
description: |
  Task: Navigate a quadruped robot through simulated tall grass terrain.
  
  Environment:
  - Room size: 18.3m × 9.1m
  - Grass height: 0.1m to 0.7m (varies by phase)
  - Grass effect: Increased ground friction, reduced visibility
  
  Robot: Boston Dynamics Spot (12 DOF quadruped)
  - Observations: position, orientation, joint states, target direction
  - Actions: forward velocity (vx), yaw rate (vyaw)
  
  Goal: Navigate from start to target position
  - Start: (1.0, 1.0, 0.8)
  - Target: (17.3, 8.1, 0.0)
  
  Challenges:
  - Grass creates resistance proportional to height
  - Tall grass can destabilize gait
  - Must maintain balance while pushing through vegetation
  
  Success criteria:
  - Reach target within 60 seconds
  - Maintain stability (no falls)
  - Minimize collisions
```

**Status**: 🔄 Pending validation

---

### BP-EUR-002: [PENDING] Reward Component Tracking

**Context**: Track which reward components Eureka generates across iterations.

**Tracking Table**:

| Iteration | Sample | Progress | Stability | Energy | Clearance | Custom | Score |
|-----------|--------|----------|-----------|--------|-----------|--------|-------|
| 1 | 1 | | | | | | |
| 1 | 2 | | | | | | |
| ... | ... | | | | | | |

**Status**: 🔄 Waiting for first Eureka run

---

### BP-EUR-003: [PENDING] Eureka Hyperparameter Selection

**Context**: Choosing appropriate Eureka settings for grass navigation.

**Recommended Settings** (to be validated):
```yaml
eureka_settings:
  model: gpt-4-turbo        # Best quality
  samples: 16               # Diverse reward candidates
  iterations: 5             # Sufficient evolution
  training_steps: 500       # Quick evaluation
  
  # For budget runs:
  model: gpt-3.5-turbo-16k
  samples: 8
  iterations: 3
  training_steps: 250
```

**Status**: 🔄 Pending validation

---

## Reward Engineering Insights

### REI-001: [PENDING] Eureka-Discovered Reward Components

**Context**: Document novel reward terms that Eureka discovers.

**Expected Components** (hypotheses):
1. **Height-adaptive clearance**: Foot lift proportional to grass height
2. **Gait frequency regulation**: Maintain optimal step frequency
3. **Forward momentum bonus**: Reward sustained forward motion
4. **Oscillation penalty**: Penalize excessive corrections

**Actual Discoveries**: TBD after first Eureka run

**Status**: 🔄 Pending

---

### REI-002: [PENDING] Reward Weight Evolution

**Context**: Track how reward weights evolve across Eureka iterations.

**Evolution Pattern**: TBD

**Status**: 🔄 Pending

---

## Performance Analysis

### PA-001: [PENDING] Baseline vs Eureka Comparison

**Metrics Table**:

| Metric | Flat Baseline | Manual R4 | Eureka Best | Δ vs Baseline |
|--------|---------------|-----------|-------------|---------------|
| TCR (H3) | ~75% | TBD | TBD | TBD |
| Nav Time | ~48s | TBD | TBD | TBD |
| Stability | ~0.78 | TBD | TBD | TBD |
| Falls/Episode | ~0.3 | TBD | TBD | TBD |

**Status**: 🔄 Pending

---

### PA-002: [PENDING] Training Efficiency

**Context**: Compare training efficiency between manual and Eureka rewards.

| Method | Iterations to 80% TCR | Wall-Clock Time | GPU Hours |
|--------|----------------------|-----------------|-----------|
| Manual R4 | TBD | TBD | TBD |
| Eureka Best | TBD | TBD | TBD |

**Status**: 🔄 Pending

---

## Phase-by-Phase Notes

### Phase E1: Environment Setup

**Date Started**: TBD
**Date Completed**: TBD

**Learnings**:
- TBD

**Issues Encountered**:
- TBD

**Resolution**:
- TBD

---

### Phase E2: Baseline Establishment

**Date Started**: TBD
**Date Completed**: TBD

**Learnings**:
- TBD

**Baseline Results**:
```
H0 (Flat): TCR = ___%, Time = ___s, Stability = ___
H1 (0.1m): TCR = ___%, Time = ___s, Stability = ___
H2 (0.3m): TCR = ___%, Time = ___s, Stability = ___
H3 (0.5m): TCR = ___%, Time = ___s, Stability = ___
H4 (0.7m): TCR = ___%, Time = ___s, Stability = ___
```

---

### Phase E3: Eureka Reward Evolution

**Date Started**: TBD
**Date Completed**: TBD

**Iteration Summaries**:

#### Iteration 1
- **Best Sample**: #___
- **Key Insight**: TBD
- **Top Reward Score**: ___

#### Iteration 2
- **Best Sample**: #___
- **Key Insight**: TBD
- **Top Reward Score**: ___

(Continue for all iterations)

---

### Phase E4: Extended Training

**Date Started**: TBD
**Date Completed**: TBD

**Training Results**:

| Policy | Reward | Final TCR | Final Time | Notes |
|--------|--------|-----------|------------|-------|
| E4-1 | Best-1 | | | |
| E4-2 | Best-1 + Curr | | | |
| E4-3 | Best-1 + Curr + DR | | | |
| E4-4 | Best-2 + Curr + DR | | | |
| E4-5 | Best-3 + Curr + DR | | | |

---

### Phase E5: Evaluation

**Date Started**: TBD
**Date Completed**: TBD

**Statistical Analysis**:
- TBD

**Key Findings**:
- TBD

---

### Phase E6: Documentation

**Date Started**: TBD
**Date Completed**: TBD

**Final Deliverables**:
- [ ] Best reward function documented
- [ ] Deployment package created
- [ ] Lessons learned finalized

---

## Common Issues & Solutions

### Issue Template

```markdown
### ISSUE-EUR-XXX: [Title]

**Date**: YYYY-MM-DD
**Severity**: Critical / High / Medium / Low

**Symptoms**:
- [What you observed]

**Root Cause**:
- [Why it happened]

**Solution**:
- [How you fixed it]

**Prevention**:
- [How to avoid in future]
```

---

## Quick Reference

### Eureka Commands

```bash
# Dry run (test prompts without training)
python run_eureka_grass.py --dry-run

# Quick test (few samples, short training)
python run_eureka_grass.py --iterations 2 --samples 4 --training-iters 100

# Basic run
python run_eureka_grass.py --iterations 5 --samples 16

# Full run (recommended)
python run_eureka_grass.py --iterations 10 --samples 32 --training-iters 1000 --num-envs 4096

# Different grass levels
python run_eureka_grass.py --grass-height H1  # Easy (0.1m)
python run_eureka_grass.py --grass-height H2  # Medium (0.3m)
python run_eureka_grass.py --grass-height H3  # Hard (0.5m) [default]
python run_eureka_grass.py --grass-height H4  # Expert (0.7m)
```

### Environment Activation

```powershell
# IsaacLab environment (Python 3.11)
conda activate isaaclab311

# Set OpenAI API key
$env:OPENAI_API_KEY = "your-key-here"
```

### Key File Locations

| File | Purpose |
|------|---------|
| `eureka/cfg/env/spot_grass_nav.yaml` | Task configuration |
| `eureka/outputs/` | Generated rewards and logs |
| `isaacgymenvs/tasks/spot_grass_nav.py` | Environment implementation |
| `experimental_design_grass/results/` | Evaluation results |

---

## Related Documents

- [Eureka Plan](./eureka_plan.md) - Implementation plan
- [Phase 5 RL](./phases/phase_5_advanced_rl.md) - Manual RL approach
- [Grass Lessons Learned](./lessons_learned.md) - General grass experiment learnings
- [Spot Training Lessons](../../../Spot-Quadruped-Training/LESSONS_LEARNED.md) - Prior Spot learnings

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-04 | AI2C Team | Initial template |

---

## Notes Section

*Use this section for quick notes during experiments:*

```
[2026-02-04] Document created, waiting to start Phase E1
```
