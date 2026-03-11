# Mason Hybrid Technical Manual

> Reference documentation for the Mason Hybrid training configuration. Covers architecture decisions, config internals, AI Coach integration, and operational procedures.
>
> **Companion doc:** `Mason_Hybrid_Training.md` (overview and rationale)
> **Parent docs:** `AI_Training_Tech_Manual.md`, `TRAINING_CURRICULUM.md`
> **Created:** March 10, 2026

---

## 1. Architecture Overview

```
Mason's Proven Foundation          Our Additions              AI Coach (Deferred)
┌─────────────────────────┐   ┌───────────────────────┐   ┌─────────────────────┐
│ 11 reward terms          │   │ terrain_relative_height│   │ Silent → Passive →  │
│ [512, 256, 128] network  │   │ dof_pos_limits         │   │ Active activation   │
│ Adaptive KL LR schedule  │   │ clamped action_smooth  │   │ Tighter bounds      │
│ Lighter DR (±2.5kg)      │   │ PhysX buffer sizing    │   │ LR/noise locked     │
│ body_contact termination │   │ Bug #22/#28b frozen    │   │ mason_hybrid_bounds  │
└─────────────┬───────────┘   └──────────┬────────────┘   └──────────┬──────────┘
              │                          │                           │
              └──────────────┬───────────┘                           │
                             ▼                                       │
                ┌────────────────────────┐                           │
                │ SpotMasonHybridEnvCfg  │ ◄─────────────────────────┘
                │ ROBUST_TERRAINS_CFG    │   (runtime weight changes via
                │ 4096 envs, 20K iters   │    env.reward_manager._term_cfgs)
                └────────────────────────┘
```

---

## 2. Environment Config (`mason_hybrid_env_cfg.py`)

### 2.1 Class Hierarchy

```
LocomotionVelocityRoughEnvCfg (Isaac Lab base)
  └── SpotMasonHybridEnvCfg
        ├── scene: HybridSceneCfg(InteractiveSceneCfg)
        ├── observations: HybridObservationsCfg
        ├── actions: HybridActionsCfg
        ├── commands: HybridCommandsCfg
        ├── rewards: HybridRewardsCfg (14 terms)
        ├── terminations: HybridTerminationsCfg
        ├── events: HybridEventCfg
        └── curriculum: HybridCurriculumCfg
```

**Critical implementation detail:** `HybridSceneCfg` must be a standalone class with terrain, robot, sensors, and lights as **class-level attributes** (not set in `__post_init__`). The parent class `LocomotionVelocityRoughEnvCfg.__post_init__()` accesses `self.scene.terrain.physics_material` before the child's `__post_init__` runs. This matches Mason's `MySceneCfg` pattern.

### 2.2 Reward Structure (14 terms)

#### Mason's 11 Original Terms

| Term | Weight | Function | Notes |
|------|--------|----------|-------|
| air_time | 5.0 | `spot_mdp.air_time_reward` | mode_time=0.3, velocity_threshold=0.5 |
| base_angular_velocity | 5.0 | `spot_mdp.base_angular_velocity_reward` | std=2.0 |
| base_linear_velocity | 5.0 | `spot_mdp.base_linear_velocity_reward` | std=1.0, ramp_rate=0.5, ramp_at_vel=1.0 |
| foot_clearance | 0.5 | `spot_mdp.foot_clearance_reward` | target_height=0.1 (ours was 0.125) |
| gait | 10.0 | `spot_mdp.GaitReward` | std=0.1, max_err=0.2, velocity_threshold=0.5 |
| action_smoothness | -1.0 | `clamped_action_smoothness_penalty` | **Clamped** (Bug #29), Mason used raw |
| air_time_variance | -1.0 | `spot_mdp.air_time_variance_penalty` | |
| base_motion | -2.0 | `spot_mdp.base_motion_penalty` | |
| base_orientation | -3.0 | `spot_mdp.base_orientation_penalty` | |
| foot_slip | -0.5 | `spot_mdp.foot_slip_penalty` | threshold=1.0 |
| joint_pos | -0.7 | `spot_mdp.joint_position_penalty` | stand_still_scale=5.0, velocity_threshold=0.5 |

#### Mason's Hip-Only Penalties

| Term | Weight | Joint Pattern | Notes |
|------|--------|--------------|-------|
| joint_acc | -1.0e-4 | `.*_h[xy]` | **Hip joints only** (not all joints like ours) |
| joint_vel | -1.0e-2 | `.*_h[xy]` | **Hip joints only** |
| joint_torques | -5.0e-4 | `.*` | All joints |

Mason applies `joint_acc` and `joint_vel` penalties only to hip joints (`_hx` and `_hy`), not knees. This gives the knee joints freedom to adapt to terrain while keeping hip movements smooth. Our config penalized all 12 joints, which over-constrained the policy.

#### Our 3 Additions

| Term | Weight | Function | Why |
|------|--------|----------|-----|
| terrain_relative_height | -2.0 | `terrain_relative_height_penalty` | Bug #27: prevents belly-crawl exploit. **Fixed 0.37m target** (v2 — was variance-based 0.35-0.42m in v1, but variable target let robot learn to crawl) |
| dof_pos_limits | -3.0 | `mdp.joint_pos_limits` | Prevents knee locking at URDF mechanical stops |
| body_height_tracking | **0.0** | (frozen) | Bug #22: world-frame Z meaningless on rough terrain |
| stumble | **0.0** | (frozen) | Bug #28b: world-frame Z misclassifies on elevated terrain |

#### Terms Dropped From Our Config

These were in our 22-term config but not in Mason's and not needed:
- `velocity_modulation` — Mason doesn't use terrain-scaled velocity commands
- `vegetation_drag` — our custom penalty, not needed with Mason's lighter DR
- `undesired_contacts` — Mason uses `body_contact` termination instead (hard kill, not soft penalty)
- `body_scraping` — redundant with `body_contact` termination
- `contact_force_smoothness` — Mason doesn't use it

### 2.3 Terminations

| Term | Function | Notes |
|------|----------|-------|
| time_out | `mdp.time_out` | 20s episodes (Mason's, ours was 30s) |
| body_contact | `mdp.illegal_contact` | Bodies: `["body", ".*leg"]`, threshold=1.0. **Hard kill** — Mason's approach (ours used soft penalty) |
| body_flip_over | `mdp.bad_orientation` | 150 degrees — our addition for extreme cases |
| terrain_out_of_bounds | `mdp.terrain_out_of_bounds` | 3.0m buffer |

### 2.4 Domain Randomization

| Parameter | Mason Hybrid | Our Config | Impact |
|-----------|-------------|------------|--------|
| Mass | ±2.5 kg | ±5.0 kg | Less sim-to-real gap noise |
| Static friction | 0.3-1.0 | 0.15-1.0 | Less extreme low-friction |
| Dynamic friction | 0.3-0.8 | 0.15-0.8 | Less extreme low-friction |
| Push velocity | ±0.5 m/s | ±0.5 m/s | Same |
| Push interval | 10-15s | 10-15s | Same |
| External forces | 0.0 | 0.0 | Disabled in both |
| Observation noise | Disabled | Enabled | Cleaner learning signal |

### 2.5 Physics and Simulation

```python
decimation = 10          # 500 Hz sim, 50 Hz control
episode_length_s = 20.0  # Mason's (ours was 30)
sim.dt = 0.002          # 500 Hz
sim.physx.gpu_collision_stack_size = 2**31     # For dense terrain
sim.physx.gpu_max_rigid_contact_count = 2**24  # For dense terrain
sim.physx.gpu_max_rigid_patch_count = 2**24    # For dense terrain
```

The PhysX GPU buffer settings are required for `ROBUST_TERRAINS_CFG` (400 patches with friction randomization). Mason's simpler terrain didn't need them — his config omits these settings. Without them, PhysX crashes with `gpu_collision_stack_size buffer overflow`.

---

## 3. PPO Config (`rsl_rl_mason_hybrid_cfg.py`)

### 3.1 Network Architecture

```
Actor:  obs(235) → [512] → [256] → [128] → actions(12)
Critic: obs(235) → [512] → [256] → [128] → value(1)
```

- **Total params:** ~800K (vs our 2.4M)
- **Activation:** ELU
- **Observation normalization:** Disabled
- **Init noise std:** 1.0 (ours was 0.5)

### 3.2 Hyperparameters

| Parameter | Value | Our Config | Notes |
|-----------|-------|------------|-------|
| num_steps_per_env | 24 | 32 | Shorter rollouts |
| num_learning_epochs | 5 | 8 | Fewer passes per batch |
| num_mini_batches | 4 | 64 | Much larger mini-batches |
| learning_rate | 1e-3 | 3e-5 (fixed) | Starts high, KL adjusts |
| schedule | `"adaptive"` | `"fixed"` + cosine | KL-based auto-adjustment |
| desired_kl | 0.01 | N/A | KL divergence target |
| value_loss_coef | 0.5 | 1.0 | Lower critic influence |
| entropy_coef | 0.0025 | 0.01 | Less exploration bonus |
| clip_param | 0.2 | 0.2 | Same |
| gamma | 0.99 | 0.99 | Same |
| lam | 0.95 | 0.95 | Same |
| max_grad_norm | 1.0 | 1.0 | Same |

### 3.3 Adaptive KL Schedule

RSL-RL's adaptive schedule adjusts LR based on KL divergence between old and new policies:
- If KL > `desired_kl * 1.5`: LR decreases (policy changing too fast)
- If KL < `desired_kl / 1.5`: LR increases (policy not learning enough)

This is why `lr_change_enabled = False` in the coach config — the adaptive schedule already manages LR, and the coach interfering would cause oscillations.

### 3.4 Mini-Batch Size Implications

Mason uses 4 mini-batches vs our 64. With 4096 envs and 24 steps/env:
- Total batch: 4096 × 24 = 98,304 samples
- Mini-batch: 98,304 / 4 = **24,576 samples**
- Our mini-batch: (5000 × 32) / 64 = **2,500 samples**

Larger mini-batches mean lower gradient noise per update step but fewer updates per epoch. Combined with 5 epochs (vs our 8), each sample is used 5 times (vs 8), reducing overfitting risk.

---

## 4. AI Coach Integration

### 4.1 Phase Config

```python
PhaseConfig(
    name="mason_hybrid",
    terrain="robust",
    num_envs=4096,
    max_iterations=20000,
    lr_max=1e-3,           # Managed by adaptive KL
    max_noise_std=1.0,     # Managed by adaptive schedule
    min_terrain_level=6.0, # Go/no-go target
    min_survival_rate=0.80,
    max_flip_rate=0.15,
    max_value_loss=15.0,
)
```

### 4.2 Disabled Coach Capabilities

| Capability | Status | Reason |
|-----------|--------|--------|
| LR changes | **Disabled** | Adaptive KL schedule manages LR |
| Noise changes | **Disabled** | Adaptive schedule manages noise |
| Weight changes | Enabled | Primary coach intervention mechanism |
| Phase advances | N/A | Single-phase run, no phase transitions |

### 4.3 Tighter Weight Bounds (`mason_hybrid_bounds`)

| Term | General Bounds | Mason Hybrid Bounds | Rationale |
|------|---------------|---------------------|-----------|
| base_linear_velocity | 1.0-15.0 | **3.0-7.0** | Prevent velocity drift (11l lesson: 5→14.26) |
| base_angular_velocity | 1.0-15.0 | **3.0-7.0** | Same |
| gait | 0.5-15.0 | **5.0-12.0** | Keep gait dominant but not overwhelming |
| joint_pos | -1.0 to -0.05 | **-1.0 to -0.3** | Never loosen below -0.3 (causes leg crossing) |
| air_time | 1.0-10.0 | **2.0-8.0** | Tighter range |
| foot_clearance | 1.0-5.0 | **0.2-1.5** | Mason's lower weight needs tighter bounds |
| action_smoothness | -5.0 to -0.05 | **-3.0 to -0.3** | Keep meaningful penalty |
| base_motion | -5.0 to -0.1 | **-4.0 to -1.0** | Don't let coach disable it |
| base_orientation | -10.0 to -0.5 | **-5.0 to -1.5** | Keep strong orientation signal |

### 4.4 Deferred Activation Implementation

In `train_ai.py`:

```python
# State variables
_coach_activated = False       # Has coach ever made an API call?
_first_plateau_seen = False    # Has passive mode seen its first plateau?

# Stage transitions:
# 1. Silent: _coach_activated=False
#    - Metrics collected, logged to JSONL
#    - No API calls, no changes
#    - Status printed every 500 iters

# 2. Passive: _coach_activated=True, coach._passive_mode=True
#    - Triggered when plateau detected AND iters > activation_threshold
#    - API calls begin, but system prompt says "RESPECT THE BASELINE"
#    - Coach biased toward no_change

# 3. Active: _coach_activated=True, coach._passive_mode=False
#    - Triggered when passive-mode coach makes its first adjust_weights
#    - Full intervention capability with mason_hybrid_bounds
```

### 4.5 Adaptive LR Detection

```python
_use_adaptive_lr = (agent_cfg.algorithm.schedule == "adaptive")

# In training loop:
if not _use_adaptive_lr:
    cosine_lr = cosine_annealing_lr(...)
    for pg in optimizer.param_groups:
        pg["lr"] = cosine_lr
# else: RSL-RL's adaptive schedule handles LR internally
```

### 4.6 Iteration Offset

When resuming from a checkpoint, `train_ai.py` uses:

```python
_resume_offset = getattr(runner, "current_learning_iteration", 0)
_iteration_counter = [_resume_offset]  # Coach reports absolute iterations
```

This ensures the decision log and TensorBoard show absolute iteration numbers, not relative-to-resume.

---

## 5. Terrain Configuration

Uses `ROBUST_TERRAINS_CFG` from `pkg/tasks/locomotion/mdp/terrains.py` — identical to Trial 11l:

- **12 terrain types:** flat, slopes (up/down), stairs (up/down), random rough, discrete obstacles, stepping stones, gaps, wave terrain, pyramids (up/down), random boulders
- **10 difficulty rows** (curriculum progression)
- **400 patches** (20 cols × 20 rows)
- **Curriculum:** `terrain_levels_vel` — promotes based on velocity + survival
- **Friction:** Randomized per-patch (ground material multiply mode)

---

## 6. Gym Registration

In `pkg/tasks/locomotion/config/spot/__init__.py`:

```python
gym.register(
    id="Locomotion-MasonHybrid-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": SpotMasonHybridEnvCfg,
        "rsl_rl_cfg_entry_point": SpotMasonHybridPPORunnerCfg,
    },
)

gym.register(
    id="Locomotion-MasonHybrid-Spot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": SpotMasonHybridEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": SpotMasonHybridPPORunnerCfg,
    },
)
```

---

## 7. Deployment

### 7.1 File Locations (H100)

```
~/multi_robot_training_new/
├── source/quadruped_locomotion/           # pip install -e .
│   └── quadruped_locomotion/
│       ├── tasks/locomotion/config/spot/
│       │   ├── mason_hybrid_env_cfg.py    # Env config
│       │   ├── __init__.py                # Gym registration
│       │   └── agents/
│       │       └── rsl_rl_mason_hybrid_cfg.py  # PPO config
│       └── ai_trainer/
│           ├── config.py                  # mason_hybrid phase + bounds
│           ├── coach.py                   # Passive mode + VLM multimodal
│           ├── guardrails.py              # LR/noise disable, tight bounds, terrain-gated penalties
│           └── prompt_builder.py          # Gait-quality-first + VLM visual analysis
├── scripts/rsl_rl/
│   └── train_ai.py                        # Entry point
└── logs/rsl_rl/spot_hybrid_ppo → ~/logs/rsl_rl/spot_hybrid_ppo  # Symlink (Bug MH-3)
```

### 7.2 Deployment Steps

```bash
# From local machine
scp -r multi_robot_training/ t2user@172.24.254.24:~/multi_robot_training_new/

# On H100
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
cd ~/multi_robot_training_new
pip install -e source/quadruped_locomotion/

# Verify
python -c "import quadruped_locomotion; import gymnasium; print(gymnasium.spec('Locomotion-MasonHybrid-Spot-v0'))"
```

### 7.3 GPU Resource Usage

With both Mason's original training and Mason Hybrid running concurrently:

| Process | VRAM | Notes |
|---------|------|-------|
| Mason's training (5000 envs, simpler terrain) | ~5.6 GB | PID 11398 |
| Mason Hybrid (4096 envs, 400-patch terrain) | ~16.6 GB | PID 20370 |
| **Total** | **~22.2 GB / 95.8 GB** | 23% utilization |

Temperature: 48°C (safe range). Throughput: ~11,500 steps/s at 8.5s/iter.

---

## 8. Known Issues and Fixes Applied

| Issue | Fix | File |
|-------|-----|------|
| `AttributeError: 'InteractiveSceneCfg' has no attribute 'terrain'` | `HybridSceneCfg` class with class-level attributes | `mason_hybrid_env_cfg.py` |
| `gpu_collision_stack_size buffer overflow` | PhysX buffer sizing (2^31, 2^24, 2^24) | `mason_hybrid_env_cfg.py` |
| Coach interfering with adaptive LR | `lr_change_enabled = False` | `config.py` |
| Coach interfering with adaptive noise | `noise_change_enabled = False` | `config.py` |
| Velocity reward drift (11l lesson) | `mason_hybrid_bounds` caps at 7.0 | `config.py` |
| Coach loosening penalties at low terrain (MH-1) | `penalty_loosen_terrain = 4.0` guardrail | `config.py`, `guardrails.py` |
| Coach can't see gait quality (MH-1) | VLM mode: `--enable_vision` sends rendered frames | `coach.py`, `train_ai.py` |
| Variable height target lets robot crawl (MH-1) | Fixed `target_height=0.37`, `terrain_scaled=False` | `mason_hybrid_env_cfg.py` |
| Coach destroys gait chasing terrain numbers (MH-1) | Gait-quality-first prompt rewrite | `prompt_builder.py` |
| Unbounded action_smoothness L2 norm | `clamped_action_smoothness_penalty` | `mason_hybrid_env_cfg.py` |
| World-frame Z height tracking | `body_height_tracking = 0.0` (frozen) | `mason_hybrid_env_cfg.py`, `guardrails.py` |
| World-frame Z stumble penalty | `stumble = 0.0` (frozen) | `mason_hybrid_env_cfg.py`, `guardrails.py` |
| Belly-crawl exploit without height penalty | `terrain_relative_height = -2.0` | `mason_hybrid_env_cfg.py` |
| TensorBoard stuck at iter 96 (Bug MH-1) | TB pointed at `~/IsaacLab/logs/` but training wrote to `~/logs/` (cwd-relative). Fix: `--logdir ~/logs/rsl_rl/spot_hybrid_ppo/` | TensorBoard launch command |
| Coach 401 auth error, disables after 3 failures (Bug MH-2) | Expired API key in `~/.anthropic_key`. `_consecutive_failures` counter can't reset without restart — must kill and resume from checkpoint | `~/.anthropic_key`, restart required |
| `FileNotFoundError` on resume — log root mismatch (Bug MH-3) | `get_checkpoint_path()` builds path from cwd. Checkpoints in `~/logs/` but cwd is `~/multi_robot_training_new/`. Fix: `ln -sf ~/logs/rsl_rl/spot_hybrid_ppo ~/multi_robot_training_new/logs/rsl_rl/spot_hybrid_ppo` | Symlink |

---

## 9. Decision Log Schema

Location: `~/logs/rsl_rl/spot_hybrid_ppo/<run_dir>/ai_coach_decisions.jsonl`

Each line is a JSON object:

```json
{
  "iteration": 500,
  "timestamp": "2026-03-10T19:30:00",
  "phase": "mason_hybrid",
  "decision": {
    "action": "no_change",
    "reasoning": "Terrain climbing steadily, metrics healthy",
    "weight_changes": {},
    "lr_change": null,
    "noise_change": null,
    "confidence": 0.85
  },
  "metrics": {
    "mean_reward": 150.5,
    "mean_terrain_level": 2.3,
    "flip_rate": 0.02,
    "value_loss": 1.8,
    "noise_std": 0.65
  },
  "guardrail_messages": [],
  "api_latency_ms": 3200
}
```

### Querying the Decision Log

```bash
# All interventions (non-no_change)
python3 -c "
import json
with open('ai_coach_decisions.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if d.get('decision',{}).get('action') != 'no_change':
            print(d['iteration'], d['decision']['action'], d['decision']['reasoning'][:80])
"

# Weight change history
python3 -c "
import json
with open('ai_coach_decisions.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if d.get('applied_changes'):
            print(f'Iter {d[\"iteration\"]}:')
            for name, (old, new) in d['applied_changes'].items():
                print(f'  {name}: {old} -> {new}')
"
```

---

## 10. Comparison With Trial 11l Final State

| Metric | Trial 11l v8 (final) | MH-1 (iter 141) | Target |
|--------|---------------------|------------------|--------|
| Terrain | 4.83 (ceiling) | 0.02 (learning) | 6.0+ |
| Reward | 640-682 | 40.1 | N/A (different scale) |
| Survival | 90.5% | 61.5% (timeout) | >80% |
| Flip rate | 9.0% | 0.0% | <15% |
| Value loss | 4.2-5.1 | 1.6 | <15 |
| Noise | 0.35 (fixed ceiling) | 0.51 (adaptive, dropping) | Auto-managed |
| Coach decisions | 67 (58 no_change, 9 adjust) | 0 (silent mode) | Minimal |
| Velocity reward | 11.41 (drifted from 5) | 5.0 (Mason's baseline) | 3.0-7.0 |
| Network params | 2.4M | 800K | — |

---

*Created for AI2C Tech Capstone — MS for Autonomy, March 2026*
