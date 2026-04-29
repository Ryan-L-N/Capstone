# Evaluated Policies

Trained locomotion policies for Boston Dynamics Spot, evaluated across 4 terrain environments (friction, grass, boulder, stairs). All policies use the Mason hybrid architecture: [512, 256, 128] actor/critic, 235-dim observations (187 height scan + 48 proprioceptive), 12-dim actions.

## Policies

| File | Description | Training |
|------|-------------|----------|
| `distilled_6899.pt` | **Best generalist** — multi-expert distillation combining friction + obstacle experts | 5K iters distillation (~8.5h) |
| `mason_hybrid_best_33200.pt` | **Best friction/grass** — Mason baseline + AI coach fine-tuning | Trial 12, 33.2K iters |
| `obstacle_best_44400.pt` | **Best boulder** — obstacle-focused training with loosened penalties | Trial 12b, 44.4K iters |
| `mason_baseline_final_19999.pt` | Mason's original baseline (no AI coach) | 20K iters |
| `hybrid_nocoach_19999.pt` | Mason architecture trained without AI coach | 20K iters |

## Evaluation Results

### 100-Episode Statistical Means (headless)

| Policy | Friction | Grass | Boulder | Stairs* |
|--------|----------|-------|---------|---------|
| **Distilled** | **49.0m** (100/100) | **28.2m** | 19.8m | 21.0m** |
| Mason Baseline | 36.9m | 29.6m | 14.4m | 10.9m |
| Mason Parallel | 48.9m (98/100) | 27.2m | 20.3m | 11.2m |
| AI Coach v8 | 37.7m | 28.7m | 20.7m | 12.0m |

*All stair results prior to 2026-03-23 are affected by a foot-clipping bug (see below).
**Stairs distilled result is from 3 episodes with the fix applied.

### Single-Episode Visual Results

| Policy | Friction | Grass | Boulder | Stairs |
|--------|----------|-------|---------|--------|
| **Distilled** | 49.5m COMPLETE | 22.2m (3/5) | 20.8m (3/5) | 17.0m (2/5) |
| Mason Hybrid Best | 49.5m COMPLETE | 49.5m COMPLETE | 21.4m (3/5) | 11.5m (2/5) |
| Obstacle Best | 42.2m (5/5) | 31.7m (4/5) | **30.4m (4/5)** | 15.7m (2/5) |
| Hybrid No-Coach | 49.5m COMPLETE | 41.2m (5/5) | 31.6m (3/5) | 12.7m (2/5) |

### Environment Zones (50m course, 5 zones of 10m each)

| Environment | Zone 1 | Zone 2 | Zone 3 | Zone 4 | Zone 5 |
|-------------|--------|--------|--------|--------|--------|
| Friction | μ=1.0 | μ=0.5 | μ=0.3 | μ=0.15 | μ=0.05 |
| Grass | Light | Medium | Dense | Very Dense | Extreme |
| Boulder | Small rocks | Medium | Large | Dense field | Extreme |
| Stairs | 3cm steps | 8cm steps | 13cm steps | 18cm steps | 23cm steps |

## How to Evaluate

```bash
# From repo root
cd Experiments/Alex/4_env_test

# Run all 4 environments, 100 episodes each (headless)
bash scripts/run_distilled_100ep.sh --headless

# Run single environment, rendered (visual)
conda activate isaaclab311
python src/run_capstone_eval.py \
    --robot spot --policy rough --env boulder --mason \
    --num_episodes 1 \
    --checkpoint policies/distilled_6899.pt

# All policies use --mason flag (obs order: height_scan first, action_scale=0.2)
```

## Architecture Details

All policies share the same architecture and are interchangeable:
- **Actor:** 235 → 512 → 256 → 128 → 12 (ELU activations)
- **Critic:** 235 → 512 → 256 → 128 → 1
- **Observations:** 187 height scan + 3 base angular velocity + 3 projected gravity + 12 joint positions + 12 joint velocities + 12 last actions + 3 velocity commands + 3 phase signals = 235
- **Actions:** 12 joint position targets (4 hx + 4 hy + 4 kn)
- **Action scale:** 0.2 (Mason baseline)
- **Control rate:** 50 Hz (decimation=10 at 500 Hz physics)

## Known Issues

- **Stairs foot clipping (fixed 2026-03-23):** Robot feet could penetrate stair step geometry and get stuck. Fixed by adding PhysX contact offset (0.02m) and raising spawn height to 0.8m for stairs. All stair results before this fix are artificially low (~50% of true performance).
- **Stair results need re-evaluation:** The 100-episode runs for Mason Parallel, AI Coach v8, and Mason Baseline were run before the fix. Their stair numbers should be ~2x higher with proper collision handling.

## Multi-Expert Distillation

The `distilled_6899.pt` policy was created by distilling two specialist experts into one generalist:

1. **Friction expert** (`mason_hybrid_best_33200.pt`) — trained on standard robust terrain
2. **Obstacle expert** (`obstacle_best_44400.pt`) — trained on 60% boulder/stairs terrain

The distillation uses height-scan-based routing: flat terrain → friction expert behavior, rough terrain → obstacle expert behavior. The student learns both via DAgger + PPO (α annealing 0.8→0.2 over 5K iterations).

See `Experiments/Alex/multi_expert_distillation/README.md` for full details.

## Credits

AI2C Tech Capstone — MS for Autonomy, March 2026.
