# NAV_ALEX — Phase C: Visual Terrain Navigation with Depth Camera

Standalone Isaac Lab extension that trains a **depth-camera navigation policy** on top of a **frozen Phase B locomotion policy**. The robot uses a 64x64 depth camera to see terrain ahead and learns to navigate mixed obstacles (boulders, stairs, rough ground) autonomously.

## Architecture

```
Depth Image (64x64)  →  CNN Encoder (128-dim)
                              ↓
                     + Proprioception (12-dim)
                              ↓
                     Nav Policy MLP [256,128] @ 10 Hz
                              ↓
                     Velocity Command [vx, vy, wz]
                              ↓
                     Frozen Loco Policy @ 50 Hz
                              ↓
                     Joint Positions (12-dim) → Robot
```

## Quick Start

```bash
# Install
cd NAV_ALEX
pip install -e source/nav_locomotion/

# Verify
python -c "import nav_locomotion; print('OK')"

# Smoke test (CPU, no Isaac Lab needed for unit tests)
python scripts/rsl_rl/smoke_test.py --skip_env

# Full smoke test (requires Isaac Lab + loco checkpoint)
python scripts/rsl_rl/smoke_test.py --headless --loco_checkpoint checkpoints/ai_coached_v8_10600.pt

# Train without coach (Phase C-0 warm-up)
python scripts/rsl_rl/train_nav.py --headless --no_wandb --no_coach \
    --loco_checkpoint checkpoints/ai_coached_v8_10600.pt \
    --num_envs 512 --max_iterations 5000

# Train with AI coach (Phase C-1)
python scripts/rsl_rl/train_nav.py --headless --no_wandb \
    --loco_checkpoint checkpoints/ai_coached_v8_10600.pt \
    --num_envs 512 --max_iterations 20000 --coach_interval 250

# Evaluate
python scripts/rsl_rl/play_nav.py \
    --nav_checkpoint logs/spot_nav_explore_ppo/model_final.pt \
    --loco_checkpoint checkpoints/ai_coached_v8_10600.pt \
    --num_envs 50 --num_episodes 100
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **RayCasterCamera** (not TiledCamera) | H100 has no Vulkan — RayCasterCamera uses PhysX raycasting, works headless |
| **30m depth range** | 10s lookahead at 3 m/s — enables route planning around obstacles |
| **End-to-end CNN** (not frozen encoder) | Learns depth features specific to terrain traversal |
| **Exploration mode** (no waypoints) | Maximize forward distance while surviving |
| **6-level curriculum** | Progressive difficulty with 8 terrain types |
| **AI Coach** (Claude Sonnet) | Runtime reward weight tuning every 250 iters |

## Project Structure

```
NAV_ALEX/
├── source/nav_locomotion/nav_locomotion/
│   ├── tasks/navigation/          # Env config, rewards, terrain
│   ├── modules/                   # CNN policy, loco wrapper, env wrapper
│   └── ai_coach/                  # Claude-powered reward tuning
├── scripts/rsl_rl/               # Training, eval, smoke test
├── docs/                         # Technical manual, training guide
├── checkpoints/                  # Frozen loco + trained nav checkpoints
└── logs/                         # TensorBoard, coach decisions (JSONL)
```

## Reward Terms (8 total)

| Term | Weight | Purpose |
|------|--------|---------|
| forward_velocity | +10.0 | World-frame forward speed |
| survival | +1.0 | Per-step alive bonus |
| terrain_traversal | +2.0 | Cumulative X-distance |
| terrain_relative_height | -2.0 | Stand upright (anti-crawl) |
| drag_penalty | -1.5 | Low height + forward = belly-sliding |
| cmd_smoothness | -1.0 | Smooth velocity commands |
| lateral_velocity | -0.3 | Light penalty for excessive strafing |
| angular_velocity | -0.5 | Prevent spinning |

## Dependencies

- Isaac Lab (isaaclab, isaaclab_tasks, isaaclab_assets)
- RSL-RL (rsl_rl)
- PyTorch
- Anthropic SDK (for AI coach only)

## H100 Deployment

```bash
# Copy to H100
scp -r NAV_ALEX/ t2user@172.24.254.24:~/NAV_ALEX/

# On H100
conda activate env_isaaclab
cd ~/NAV_ALEX
pip install -e source/nav_locomotion/
pip install anthropic  # For AI coach

# Copy loco checkpoint
cp ~/multi_robot_training/checkpoints/ai_coached_v8_10600.pt checkpoints/

# Train
python scripts/rsl_rl/train_nav.py --headless --no_wandb \
    --loco_checkpoint checkpoints/ai_coached_v8_10600.pt \
    --num_envs 512 --coach_interval 250
```
