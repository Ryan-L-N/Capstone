# Phase C Navigation — Technical Manual

Authoritative reference for the hierarchical visual terrain navigation system. Covers the full pipeline from depth camera input to joint-level actuation, including the AI coach, reward system, terrain curriculum, and all lessons inherited from Phase B.

---

## 1. Architecture Overview

Phase C uses a three-tier hierarchical control architecture:

```
Nav Policy (10 Hz)     ->  Frozen Loco Policy (50 Hz)  ->  Physics (500 Hz)
[depth + proprio]          [vel_cmd -> joints]              [PhysX GPU solver]
  ActorCriticCNN             FrozenLocoPolicy                Isaac Lab env
  obs: 4108 dim              obs: 235 dim                    dt: 0.002s
  act: 3 dim (vel_cmd)       act: 12 dim (joints)            decimation: 50
```

The nav policy sees a 64x64 depth image and 12-dim proprioception, outputs 3-dim velocity commands `[vx, vy, wz]`, which the frozen loco policy translates to 12-dim joint position targets. The `NavEnvWrapper` orchestrates the handoff. RSL-RL's `OnPolicyRunner` sees only the nav-level interface: 4108-dim observations, 3-dim actions.

**Rate hierarchy:** The env `decimation=50` means 50 physics steps (at 500 Hz) per nav step, giving a 10 Hz nav policy. The frozen loco policy runs once per nav step (velocity commands held constant for the full 0.1s nav interval). Physics sub-stepping at 500 Hz handles contact dynamics.

---

## 2. Depth Camera System

### Why RayCasterCamera (Not TiledCamera)

The H100 training server runs headless without Vulkan/display. `TiledCamera` requires a GPU-accelerated rendering pipeline (RTX renderer), which is unavailable headless. `RayCasterCamera` uses PhysX Warp mesh raycasting against the static terrain mesh — pure compute, no rendering stack.

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Resolution | 64 x 64 | 4096 pixels, tractable for CNN without pooling bloat |
| FOV | ~90 degrees | Wide enough for obstacle detection, `horizontal_aperture=20.955mm` at `focal_length=24mm` |
| Range | 30 m | Long-range pathfinding on 50m terrain patches |
| Update rate | 10 Hz (`update_period=0.1`) | Matches nav policy frequency |
| Pattern | `PinholeCameraPatternCfg` | Standard pinhole projection model |
| Mount | +0.3m X, +0.3m Z on `body` | Forward-facing, above body center |
| Tilt | 10 degrees downward | Half-angle quaternion: `(cos(-5deg), 0, sin(-5deg), 0)` |
| Data type | `distance_to_image_plane` | Per-pixel depth, clipped at `max_distance=30m` |
| Mesh target | `/World/ground` | Rays only hit terrain mesh (not robot self) |

Depth values are normalized to `[0, 1]` by the observation term (`normalize=True`). Pixels beyond 30m read as 1.0.

### Limitation

`RayCasterCamera` only raycasts against static meshes listed in `mesh_prim_paths`. Dynamic obstacles (other robots, thrown objects) are invisible to the depth camera. This is acceptable for single-robot terrain navigation but would need `TiledCamera` for multi-agent scenarios.

---

## 3. CNN Architecture

### DepthCNN Encoder

Three convolutional layers reduce the 64x64 depth image to a 128-dim feature vector:

```
Input:  (N, 1, 64, 64)
Conv2d(1, 32, kernel_size=5, stride=2) + ELU  -> (N, 32, 30, 30)
Conv2d(32, 64, kernel_size=3, stride=2) + ELU -> (N, 64, 14, 14)
Conv2d(64, 64, kernel_size=3, stride=2) + ELU -> (N, 64, 6, 6)
Flatten                                        -> (N, 2304)
Linear(2304, 128) + ELU                        -> (N, 128)
```

Flat size (2304) is computed dynamically via a dummy forward pass at init, so the encoder adapts if `depth_res` changes.

### ActorCriticCNN Policy

```
Observations (N, 4108)
    |
    +--[0:4096]-- DepthCNN --> (N, 128)  [shared backbone]
    |
    +--[4096:4108]-- Proprioception --> (N, 12)
    |
    Concatenate --> (N, 140)
        |
        +-- Actor MLP [256, 128] --> (N, 3)   [vx, vy, wz]
        |
        +-- Critic MLP [256, 128] --> (N, 1)  [value]
```

The CNN backbone is shared between actor and critic (same `DepthCNN` instance). Action noise is a learnable `nn.Parameter` with `_sanitize_std()` guarding against NaN/Inf (Bug #24 convention). Initial noise std: 0.5.

Implements the full RSL-RL `ActorCritic` interface: `act()`, `act_inference()`, `evaluate()`, `reset()`.

---

## 4. Observation Space

Total: **4108 dimensions**.

| Index Range | Component | Dims | Notes |
|-------------|-----------|------|-------|
| `[0:4096]` | Depth image (flattened) | 4096 | 64x64, normalized [0,1], 30m range |
| `[4096:4099]` | Body linear velocity | 3 | Body-frame, noise std=0.15 |
| `[4099:4102]` | Body angular velocity | 3 | Body-frame, noise std=0.15 |
| `[4102:4105]` | Projected gravity | 3 | Noise std=0.05 |
| `[4105:4108]` | Previous action (vel_cmd) | 3 | No noise |

Observation noise is applied only to proprioception (not depth). The depth image already has implicit noise from terrain mesh resolution and raycasting quantization.

---

## 5. Action Space

3 dimensions: velocity commands routed through the frozen loco policy.

| Index | Component | Range | Unit |
|-------|-----------|-------|------|
| 0 | `vx` (forward/back) | [-1.0, 3.0] | m/s |
| 1 | `vy` (left/right) | [-1.5, 1.5] | m/s |
| 2 | `wz` (yaw turn) | [-2.0, 2.0] | rad/s |

Asymmetric `vx` range: backward is limited to 1.0 m/s (retreat), forward up to 3.0 m/s (fast traversal). Full lateral control (`vy`) enables obstacle avoidance maneuvers. Commands are hard-clipped by `NavEnvWrapper._clip_vel_cmd()` before reaching the loco policy.

---

## 6. Frozen Loco Policy

### Architecture Auto-Detection

`FrozenLocoPolicy` reads the first actor layer's weight shape to determine the checkpoint architecture:

| First Hidden Dim | Full Architecture | Source |
|-------------------|-------------------|--------|
| 512 | [512, 256, 128] | Hybrid no-coach (MH-2a) |
| 1024 | [1024, 512, 256] | AI-coached v8 (Trial 11l) |

### Loading and Inference

1. Load checkpoint, extract `actor.*` keys only (critic discarded).
2. Build `nn.Sequential` matching detected architecture (Linear + ELU layers).
3. `load_state_dict(strict=True)` — architecture must match exactly.
4. Freeze all parameters (`requires_grad=False`), set to `eval()` mode.

### Velocity Command Injection

The nav policy's 3-dim output is injected into the loco observation vector at indices `[9, 10, 11]`:

```
Loco obs layout (235 dims):
  [0:3]   base_lin_vel
  [3:6]   base_ang_vel
  [6:9]   projected_gravity
  [9:12]  velocity_commands  <-- nav policy output injected here
  [12:24] joint_pos (relative to default)
  [24:36] joint_vel
  [36:48] last_action
  [48:235] height_scan (187 rays)
```

Forward pass is wrapped in `@torch.no_grad()`. The loco policy never trains; it is a frozen inference-only module.

---

## 7. NavEnvWrapper

The wrapper sits between RSL-RL and Isaac Lab, translating the 3-dim nav action space to the 12-dim loco action space.

### Step Flow

```python
NavEnvWrapper.step(vel_cmd)          # vel_cmd: (N, 3)
    -> _clip_vel_cmd(vel_cmd)        # Enforce action ranges
    -> _get_loco_obs()               # Build 235-dim loco observation
    -> loco_policy(loco_obs, vel_cmd) # Frozen inference -> 12 joint actions
    -> env.step(joint_actions)       # Isaac Lab physics step (decimation=50)
    -> _build_nav_obs()              # Build 4108-dim nav observation
    -> return (nav_obs, reward, done, info)
```

### Key Features

- **Velocity clipping:** Hard bounds on `[vx, vy, wz]` before loco injection.
- **Episode tracking:** Tracks start X-position for terrain traversal reward. Resets on termination.
- **Reward manager passthrough:** Exposes `env.reward_manager` for the AI coach to modify weights at runtime via `env.reward_manager._term_cfgs[name].weight`.
- **RSL-RL compatibility:** Exposes `num_obs=4108`, `num_actions=3`, `num_envs`, `device`, `max_episode_length`, `extras`.

---

## 8. Reward System

8 terms, all following the Bug #29 clamping convention: `compute -> clamp(lo, hi) -> nan_to_num -> isfinite guard`.

| # | Term | Weight | Purpose | Clamp Range | Type |
|---|------|--------|---------|-------------|------|
| 1 | `forward_velocity` | +10.0 | World-frame +X speed | [-1.0, 3.0] | Incentive |
| 2 | `survival` | +1.0 | Per-step alive bonus | N/A (constant 1.0) | Incentive |
| 3 | `terrain_traversal` | +2.0 | Cumulative X-distance / 50m | [0.0, 1.0] | Incentive |
| 4 | `terrain_relative_height` | -2.0 | Height deviation from target | [0.0, 1.0] | Penalty |
| 5 | `drag_penalty` | -1.5 | Low height + forward vel (anti-crawl) | [0.0, 3.0] | Penalty |
| 6 | `cmd_smoothness` | -1.0 | L2 norm of vel_cmd delta | [0.0, 5.0] | Penalty |
| 7 | `lateral_velocity` | -0.3 | vy squared (light, for obstacle avoidance) | [0.0, 2.0] | Penalty |
| 8 | `angular_velocity` | -0.5 | wz squared (anti-spinning) | [0.0, 3.0] | Penalty |

### Anti-Crawl System

Bug #27 showed that robots exploit belly-crawling when height tracking is disabled. Two terms work together:

1. **`terrain_relative_height`** — Uses height scanner center ray for ground-relative body height. When `terrain_scaled=True`, the target height adapts via height scan variance: flat terrain (variance ~0.001) targets 0.42m, rough terrain (variance ~0.02) targets 0.35m, interpolated between. Squared error is clamped to [0.0, 1.0] before squaring (Bug #28c).

2. **`drag_penalty`** — Activates when body height < 0.25m AND forward velocity > 0. Penalty = `(threshold - height) * forward_speed`, catching the specific exploit of dragging the belly forward.

---

## 9. Terrain Curriculum

6 difficulty levels x 10 terrain type columns = 60 terrain patches. Each patch is 50m long x 20m wide.

### Terrain Types

| Type | Proportion | Level 1 (Easiest) | Level 6 (Hardest) |
|------|-----------|--------------------|--------------------|
| flat | 10% | Smooth plane | Smooth plane |
| random_rough | 15% | 0.02m grid bumps | 0.12m grid bumps |
| boxes | 15% | 10 small boxes (2-8cm) | 30 large boxes (8-25cm) |
| stairs_up | 12% | 3cm step height | 20cm step height |
| stairs_down | 8% | 3cm step height | 20cm step height |
| wave | 10% | 5cm amplitude | 15cm amplitude |
| discrete_obstacles | 10% | 5cm obstacles | 20cm obstacles |
| boulders | 20% | 0.1m grid height | 0.8m grid height |

Boulders have the highest proportion (20%) because they are the primary terrain requiring visual navigation decisions — the robot must learn to see and route around large obstacles.

### Curriculum Mechanics

- All robots start at level 1 (`max_init_terrain_level=0`).
- Promotion/demotion via `mdp.terrain_levels_vel` based on survival rate and forward velocity.
- `TerrainGeneratorCfg` with `curriculum=True` enables automatic level tracking.
- Horizontal scale: 0.1m, vertical scale: 0.005m, slope threshold: 0.75.

---

## 10. AI Coach Integration

Claude Sonnet is consulted every 250 iterations (configurable via `--coach_interval`) to analyze training metrics and recommend reward weight adjustments.

### Pipeline

```
MetricsCollector.collect()
    -> Guardrails.check_emergency()     [NaN -> halt, value_loss > 100 -> halve LR]
    -> Coach.get_decision()             [Claude Sonnet API call]
    -> Guardrails.validate_weight_changes()
    -> Actuator.apply_weight_changes()  [Live env modification]
    -> DecisionLog.log_decision()       [JSONL audit trail]
```

### Weight Bounds Table

| Term | Min | Max |
|------|-----|-----|
| `forward_velocity` | 3.0 | 15.0 |
| `survival` | 0.5 | 3.0 |
| `terrain_traversal` | 0.5 | 5.0 |
| `terrain_relative_height` | -5.0 | -1.0 |
| `drag_penalty` | -4.0 | -0.5 |
| `cmd_smoothness` | -3.0 | -0.1 |
| `lateral_velocity` | -1.5 | -0.05 |
| `angular_velocity` | -2.0 | -0.1 |

### Guardrail Rules

1. **Max 3 weight changes per consultation** (Trial 11k: 6 simultaneous changes collapsed the policy to 88% flip).
2. **Max 20% delta per weight** — prevents catastrophic reward landscape shifts.
3. **Sign constraints** — positive rewards cannot become penalties and vice versa.
4. **Terrain-gated penalty loosening** — penalties cannot be loosened (made less negative) until mean terrain level >= 3.0. This prevents premature relaxation on easy terrain.
5. **Absolute bounds** — hard min/max per weight (see table above).
6. **Emergency checks** — NaN in policy params triggers immediate halt. Value loss > 100 triggers automatic LR halving (Bug #25 watchdog).

### Anti-Crawl Prompt Rule

The system prompt instructs the coach to never reduce `terrain_relative_height` or `drag_penalty` below their minimum bounds. Bug #27 demonstrated that belly-crawling emerges rapidly once height enforcement weakens.

### Cost Estimate

~$2-4 per full training run. Sonnet at ~$0.01/call, ~300 consultations per phase (20000 iters / 250 interval, plus some no-ops).

---

## 11. Training Pipeline

### Phase C-0: Flat Warm-Up (No Coach)

Verify that the CNN trains, the loco wrapper works, and the robot moves forward on flat ground. No terrain curriculum, no coach.

```bash
python scripts/rsl_rl/train_nav_no_coach.py \
    --headless --no_wandb \
    --loco_checkpoint checkpoints/ai_coached_v8_10600.pt \
    --num_envs 512 --max_iterations 5000 \
    --save_interval 100
```

**Success criteria:** Robot achieves positive forward velocity, reward trends upward, no NaN.

### Phase C-1: Full Curriculum + Coach

Enable terrain curriculum and AI coach. This is the main training phase.

```bash
python scripts/rsl_rl/train_nav.py \
    --headless --no_wandb \
    --loco_checkpoint checkpoints/ai_coached_v8_10600.pt \
    --num_envs 512 --max_iterations 20000 \
    --coach_interval 250 --save_interval 100
```

**Resume from C-0 checkpoint:**

```bash
python scripts/rsl_rl/train_nav.py \
    --headless --no_wandb \
    --loco_checkpoint checkpoints/ai_coached_v8_10600.pt \
    --num_envs 512 --max_iterations 20000 \
    --coach_interval 250 --save_interval 100 \
    --resume logs/spot_nav_explore_ppo/<timestamp>/model_5000.pt
```

### Phase C-2: Harder Terrain (Optional)

If the robot masters level 6, increase terrain difficulty parameters (boulder height, stair step height) and continue training. This phase is defined by modifying `terrains.py` parameters, not a separate script.

### H100 Deployment

```bash
scp -r NAV_ALEX/ t2user@172.24.254.24:~/NAV_ALEX/
ssh t2user@172.24.254.24
conda activate env_isaaclab
cd ~/NAV_ALEX && pip install -e source/nav_locomotion/
pip install anthropic  # For AI coach
python -c "import nav_locomotion"  # Verify gym envs register
```

---

## 12. Known Risks and Mitigations

### RayCasterCamera Static Mesh Limitation

`RayCasterCamera` only sees meshes listed in `mesh_prim_paths=["/World/ground"]`. Dynamic objects (other robots, falling debris) are invisible. **Mitigation:** Single-robot training only. Multi-agent would require `TiledCamera` with Vulkan.

### CNN Gradient Issues

The CNN backbone processes 4096-dim input through 3 conv layers with ~100K parameters. Gradient pathology (vanishing/exploding) can stall learning. **Mitigation:** ELU activations (no dead neurons), `max_grad_norm=1.0` in PPO config, learning rate 1e-4 (lower than Phase B's 3e-5 to account for CNN sensitivity).

### Belly-Crawl Exploit

Bug #27 proved robots will belly-crawl when height enforcement weakens. **Mitigation:** Two-term anti-crawl system (`terrain_relative_height` + `drag_penalty`) with guardrail-enforced minimum weights. Coach cannot reduce these below -1.0 and -0.5 respectively.

### Coach Over-Tuning

The AI coach can chase short-term metrics, causing oscillation. **Mitigation:** 20% delta cap, 3-change limit, terrain-gated loosening, and the coach sees its own recent decisions to avoid flip-flopping.

### H100 Headless Constraints

Must use `AppLauncher(headless=True)` on H100. Must pass `--headless` flag. Never call `simulation_app.close()` — use `os._exit(0)`. D-state zombie processes from CUDA deadlock require BMC reboot.

### Value Loss Explosion

Phase B showed that LR too high + small batch size causes value loss cascade: 1.0 -> 46M -> 677Q -> NaN in 3 iterations. **Mitigation:** Fixed LR at 1e-4 (not cosine annealing), `use_clipped_value_loss=True`, emergency watchdog halves LR when value_loss > 100.

---

## 13. Bug Museum References

All Bug Museum entries from Phase B that apply to Phase C:

| Bug | Title | Phase C Relevance |
|-----|-------|-------------------|
| #22 | World-frame Z height penalty on rough terrain | `terrain_relative_height_penalty` uses ground-relative Z via height scanner center ray |
| #24 | `clamp_()` does not fix NaN | `ActorCriticCNN._sanitize_std()` uses explicit NaN/Inf detection before clamping |
| #25 | Value loss oscillation cascade | Emergency watchdog in guardrails: value_loss > 100 triggers LR halving |
| #27 | Belly-crawl exploit | Two-term anti-crawl: `terrain_relative_height` + `drag_penalty` with minimum weight bounds |
| #28c | Unbounded squared height error | Height error clamped to [0.0, 1.0] before squaring |
| #29 | Unbounded penalty terms (L2 norms) | All 8 reward terms use `_safe_clamp()`: clamp + nan_to_num + isfinite guard |
| #30 | Stale critic from reward weight changes | Fresh critic at init (CNN policy built from scratch). Coach weight changes are small (20% max) to avoid stale-critic effects |
| #33 | RSL-RL internal metrics not captured | `runner.log()` interception in `train_nav.py` captures `mean_reward`, `value_loss`, `noise_std` |

### Additional Phase C Conventions

- **Never call `simulation_app.close()`** — always `os._exit(0)`.
- **Parse CLI args before Isaac Lab imports** — `AppLauncher` requirement.
- **`--headless` required on H100** after BMC reboot (Bug #28).
- **`--save_interval 100`** — 500 is too large; 65M steps between saves is the acceptable maximum.
- **Always verify deployed code matches local code** — Phase B lost days to stale H100 deployments.

---

## File Index

All paths relative to `NAV_ALEX/`:

| File | Purpose |
|------|---------|
| `source/nav_locomotion/nav_locomotion/modules/depth_cnn.py` | DepthCNN encoder + ActorCriticCNN policy |
| `source/nav_locomotion/nav_locomotion/modules/loco_wrapper.py` | FrozenLocoPolicy (Phase B checkpoint loader) |
| `source/nav_locomotion/nav_locomotion/modules/nav_env_wrapper.py` | NavEnvWrapper (vel_cmd -> joints translation) |
| `source/nav_locomotion/nav_locomotion/tasks/navigation/mdp/rewards.py` | 8 reward terms with Bug #29 clamping |
| `source/nav_locomotion/nav_locomotion/tasks/navigation/mdp/terrains.py` | 6-level curriculum terrain config |
| `source/nav_locomotion/nav_locomotion/tasks/navigation/config/spot/nav_env_cfg.py` | SpotNavExploreCfg (full env config) |
| `source/nav_locomotion/nav_locomotion/tasks/navigation/config/spot/agents/rsl_rl_ppo_cfg.py` | PPO hyperparameters |
| `source/nav_locomotion/nav_locomotion/ai_coach/coach.py` | Claude Sonnet decision engine |
| `source/nav_locomotion/nav_locomotion/ai_coach/guardrails.py` | Safety validation (weight bounds, emergency checks) |
| `scripts/rsl_rl/train_nav.py` | Main training entry point (with coach) |
| `scripts/rsl_rl/train_nav_no_coach.py` | Baseline training (no coach) |
| `scripts/rsl_rl/play_nav.py` | Evaluation / visualization |
| `scripts/rsl_rl/smoke_test.py` | Local verification |
