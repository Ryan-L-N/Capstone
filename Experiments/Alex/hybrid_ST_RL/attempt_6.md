# Attempt 6: From-Scratch Training — Fix the Stall

## Why Attempt 6?

Attempt 5 (running as "Attempt 6" on the H100) took the right strategic approach — train from scratch with terrain curriculum instead of fine-tuning. But it's stalling:

| Metric | At iter 1,086 (3.7 hours in) | Expected by this point |
|--------|------------------------------|----------------------|
| Mean reward | 1.06 (flat, not climbing) | 3–5 (climbing) |
| Episode length | 6.8s (dying every ~6s) | 15–20s (learning to stand) |
| Body contact % | 100% (every episode) | 70–80% (improving) |
| Terrain level | 0.0 (stuck on flat) | 0.5–1.0 (starting to promote) |
| Noise std | 0.30 (clamped at floor) | 0.6–0.8 (converging naturally) |

### Comparison with Published Results

Kumar et al. (2023) — "Enhancing Efficiency of Quadrupedal Locomotion over Challenging Terrains with Extensible Feet" — trained Spot from scratch in Isaac Gym with nearly identical hyperparameters (PPO, [512,256,128], ELU, adaptive LR, gamma=0.99, lambda=0.95, 5 epochs). Their reward curve (Fig. 3) shows clear upward movement well before 1,000 iterations. Key differences from our Attempt 5:

| Parameter | Kumar et al. | Our Attempt 5 | Impact |
|-----------|-------------|---------------|--------|
| Obs normalization | Standard (ON) | **OFF** | Critical — heterogeneous scales kill learning |
| Reward terms | ~8–10 | **19** | Noisy gradient signal for critic |
| Init noise std | Standard (0.3–0.5) | **1.0** | Robot flails wildly from iter 0 |
| Termination | Base contact only | **Body + all leg segments** | Robot gets killed for normal shin scraping |
| Spawn perturbation | Modest | **Aggressive** (±1.5 m/s, ±0.7 rad/s) | Robot spawns mid-tumble |
| Envs | 2,048 | 16,384 | Not a problem, just more parallelism |

## Root Cause Analysis

Five issues compound to prevent learning:

### 1. Observation Normalization Disabled (Critical)
`scratch_ppo_cfg.py:43-44` — `actor_obs_normalization=False`, `critic_obs_normalization=False`

The 235-dim observation vector mixes:
- Base linear velocity: ~0–3 m/s
- Joint velocities: ~0–10 rad/s
- Height scan: ~-1 to 1 (clipped meters)
- Gravity vector: ~0–1
- Last actions: ~-0.25 to 0.25 (after scaling)

Without running mean/std normalization, the first network layers are trying to learn features across inputs that differ by 10–40x in scale. The critic can't form useful value estimates, so advantage calculation is noisy, and the actor gets bad gradient signals.

RSL-RL's `EmpiricalNormalization` normalizes observations to zero-mean unit-variance using a running estimate. Every standard Isaac Gym locomotion paper (Rudin et al. 2021, Lee et al. 2020, Miki et al. 2022) uses this. It was likely disabled during our fine-tuning attempts (where the checkpoint's normalization stats would conflict) and never re-enabled for scratch.

### 2. init_noise_std=1.0 Too Aggressive
`scratch_ppo_cfg.py:42` — `init_noise_std=1.0`

With random network weights outputting ~N(0,1) and noise_std=1.0 on top, the effective action distribution per joint is ~N(0, sqrt(2)) * 0.25 (action_scale). This produces wild random joint targets every step. The robot doesn't get a chance to discover that "doing nothing" (staying near default joint positions) is better than flailing.

At 1.0, the noise std hits the 0.3 floor clamp almost immediately (line 46: `min_noise_std=0.3`), meaning adaptive KL drove it down aggressively — the algorithm itself is signaling that 1.0 was too high. Starting at 0.5 gives a smoother convergence path.

### 3. Termination on All Leg Segments
`finetune_env_cfg.py:422` — `body_names=["body", ".*leg"]`, threshold=1.0 N

This terminates the episode when ANY leg link (thigh, shin, knee) contacts the terrain with >1 N of force. For a robot learning from scratch, shin scraping during early stumbling is inevitable and actually informative — the robot needs those extra timesteps to learn "what went wrong" before the fall.

Kumar et al. only terminate on **base (torso) collision**. Leg contact is penalized via rewards (stumble penalty) but doesn't kill the episode. This gives the robot 2–5x more timesteps per episode to learn from.

### 4. Aggressive Spawn Perturbations
`finetune_env_cfg.py:206-216`:
```
velocity_range: x=±1.5, y=±1.0, z=±0.5
angular:        roll=±0.7, pitch=±0.7, yaw=±1.0
```

Spawning robots with ±1.5 m/s lateral velocity and ±0.7 rad/s roll/pitch means many robots start already falling. This is great for a trained policy (robustness testing) but terrible for a random-init policy that can't even stand from rest.

### 5. 19 Reward Terms Create Noisy Gradients
The 5 custom terms (vegetation_drag, velocity_modulation, body_height_tracking, contact_force_smoothness, stumble) add complexity that's irrelevant when the robot can't stand:

- **vegetation_drag** (weight -0.001): Near-zero signal, adds noise
- **velocity_modulation** (weight 2.0): Requires the robot to be moving to produce signal
- **body_height_tracking** (weight -2.0): Robot falls before height matters
- **contact_force_smoothness** (weight -0.5): Irrelevant while falling
- **stumble** (weight -2.0): Coupled with aggressive leg termination, redundant

The critic has to model a 19-term value function from iter 0 with no normalization. Simplifying to the ~14 core terms reduces noise in advantage estimates during the critical first 1,000 iterations.

## Changes for Attempt 6

### Change 1: Enable Observation Normalization
**File:** `configs/scratch_ppo_cfg.py` (lines 43-44)
```python
# BEFORE (Attempt 5):
actor_obs_normalization=False,
critic_obs_normalization=False,

# AFTER (Attempt 6):
actor_obs_normalization=True,
critic_obs_normalization=True,
```

### Change 2: Reduce Initial Noise Std
**File:** `configs/scratch_ppo_cfg.py` (line 42)
```python
# BEFORE (Attempt 5):
init_noise_std=1.0,

# AFTER (Attempt 6):
init_noise_std=0.5,
```

### Change 3: Relax Termination — Body Only
**New file:** `configs/scratch_env_cfg_v2.py` overrides `SpotFinetuneTerminationsCfg` with a scratch-specific version that only terminates on torso contact:
```python
body_contact = DoneTerm(
    func=mdp.illegal_contact,
    params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body"]),
        "threshold": 1.0,
    },
)
```
Leg contact is still penalized via the stumble reward (weight -2.0) but doesn't end the episode. The robot gets more timesteps to learn.

### Change 4: Reduce Spawn Perturbations
**New file:** `configs/scratch_env_cfg_v2.py` overrides `SpotFinetuneEventCfg` with gentler reset conditions:
```python
reset_base = EventTerm(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        "velocity_range": {
            "x": (-0.5, 0.5),       # Was ±1.5
            "y": (-0.5, 0.5),       # Was ±1.0
            "z": (-0.3, 0.3),       # Was ±0.5
            "roll": (-0.3, 0.3),    # Was ±0.7
            "pitch": (-0.3, 0.3),   # Was ±0.7
            "yaw": (-0.5, 0.5),     # Was ±1.0
        },
    },
)

reset_robot_joints = EventTerm(
    func=spot_mdp.reset_joints_around_default,
    mode="reset",
    params={
        "position_range": (-0.1, 0.1),    # Was ±0.2
        "velocity_range": (-1.5, 1.5),    # Was ±2.5
        "asset_cfg": SceneEntityCfg("robot"),
    },
)
```

### Change 5: Disable Niche Reward Terms
**New file:** `configs/scratch_env_cfg_v2.py` overrides `SpotFinetuneRewardsCfg` to zero out the 5 niche terms:
```python
vegetation_drag.weight = 0.0          # Was -0.001
velocity_modulation.weight = 0.0      # Was 2.0
body_height_tracking.weight = 0.0     # Was -2.0
contact_force_smoothness.weight = 0.0 # Was -0.5
stumble.weight = 0.0                  # Was -2.0
```
This leaves 14 active reward terms — the same core set proven in the 48hr policy. The niche terms can be re-enabled later once the robot can walk (e.g., via progressive reward scheduling, or in a Stage 2 fine-tune on top of this checkpoint).

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `configs/scratch_ppo_cfg.py` | Modify | Enable normalization, reduce noise std |
| `configs/scratch_env_cfg_v2.py` | **Create** | New env config with relaxed termination, gentler resets, simplified rewards |
| `train_from_scratch.py` | Modify | Import `SpotScratchEnvCfgV2`, update banner to say "ATTEMPT 7" |
| `scripts/train_scratch_h100.sh` | Modify | Update banner/log filename |

The finetune configs are NOT touched — all changes are isolated to scratch-specific files.

## Expected Training Trajectory (Revised)

With these fixes, learning should progress faster because:
- Normalized observations give the critic clean inputs from iter 0
- Lower noise std means less random flailing, faster convergence to standing
- Body-only termination gives 2–5x longer episodes for learning
- Gentler spawns mean the robot starts from a survivable state
- 14 reward terms give clearer gradient signal

| Phase | Iterations | Expected Behavior |
|-------|-----------|-------------------|
| Standing | 0–300 | Robot learns to stay upright, episode length 2s → 15s |
| Basic gait | 300–1,500 | Robot learns to trot, episode length 15s → 30s (timeout) |
| Terrain promotion | 1,500–5,000 | Curriculum promotes to level 1–3, terrain_levels > 0 |
| Moderate terrain | 5,000–10,000 | Handles uneven ground, levels 3–6 |
| Hard terrain | 10,000–15,000 | DR fully expanded, levels 7–9, robust policy |

## Success Criteria

- [ ] Episode length > 10s within 500 iterations (1.5 hours)
- [ ] Body contact termination < 70% within 1,000 iterations (3 hours)
- [ ] Terrain levels > 0 within 2,000 iterations (6 hours)
- [ ] Mean reward steadily climbing (not flat like Attempt 5)
- [ ] Noise std converges to 0.4–0.7 range naturally (not clamped at floor)

## Tradeoff vs Attempt 5

We trade some late-stage robustness for early-stage learnability:
- **Relaxed termination** means the final policy may be less robust to leg impacts — can be tightened later by re-enabling `".*leg"` termination and fine-tuning for 1–2K more iterations
- **Disabled niche rewards** means no vegetation/height/smoothness shaping early on — re-enable in a second phase once walking is stable
- **Gentler spawns** reduce early robustness to perturbations — progressive DR still ramps up push/force over 15K iterations

This is the standard approach in the literature: learn to walk first, then harden.
