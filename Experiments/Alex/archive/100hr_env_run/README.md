# 100hr Multi-Terrain Robust Locomotion Training

Train a single unified Spot locomotion policy that handles rough terrain, stairs (up AND down), boulders, low-friction surfaces (ice/grass/mud), and unstructured terrain.

**Target:** 100 hours on H100 NVL 96GB, ~39 billion timesteps, 60,000 PPO iterations.

## What's Different from the 48hr Run

| Parameter | 48hr Run | 100hr Run |
|---|---|---|
| Terrain types | 6 | **12** (added gaps, stepping stones, waves, rails, etc.) |
| Terrain patches | 200 (10x20) | **400 (10x40)** |
| Friction range | [0.5, 1.25] | **[0.05, 1.5]** (covers ice to high-grip) |
| Friction buckets | 64 | **256** |
| Mass offset | ±5 kg | **±8 kg** |
| Push velocity | ±0.5 m/s | **±1.5 m/s** |
| External force | ±3 N | **±8 N** |
| Network | [512, 256, 128] | **[1024, 512, 256]** (~2M params) |
| Parallel envs | 8,192 | **20,480** |
| Steps/iteration | 196,608 | **655,360** |
| Episode length | 20s | **30s** |
| Reward terms | 14 | **18** (4 new terms) |
| LR schedule | Adaptive KL | **Cosine annealing** (1e-3 → 1e-5) |

## Quick Start

### Local Debug (RTX 2000 Ada, ~5 min)
```bash
cd /c/IsaacLab
bash /path/to/100hr_env_run/scripts/train_local_debug.sh
```

### H100 Full Run (~100 hours)
```bash
ssh ai2ct2
cd ~/IsaacLab
nohup bash /path/to/100hr_env_run/scripts/train_h100.sh > ~/100hr_training.log 2>&1 &
```

### Resume from Checkpoint
```bash
bash /path/to/100hr_env_run/scripts/resume_training.sh
```

### Monitor Training
```bash
# Live logs
tail -f ~/100hr_training.log

# TensorBoard
tensorboard --logdir logs/rsl_rl/spot_100hr_robust --bind_all
```

## File Structure

```
100hr_env_run/
├── train_100hr.py              # Main training script (run this)
├── configs/
│   ├── terrain_cfg.py          # 12-type terrain generator (ROBUST_TERRAINS_CFG)
│   ├── env_cfg.py              # Environment config (rewards, DR, curriculum)
│   ├── ppo_cfg.py              # PPO hyperparameters (20K envs, cosine LR)
│   └── __init__.py
├── rewards/
│   ├── reward_terms.py         # 4 new custom reward functions
│   └── __init__.py
├── scripts/
│   ├── train_h100.sh           # H100 launch script
│   ├── train_local_debug.sh    # Local debug (64 envs, 100 iters)
│   └── resume_training.sh      # Checkpoint resume
├── checkpoints/                # Model checkpoints (saved by RSL-RL)
├── logs/                       # TensorBoard logs
└── eval/                       # Post-training evaluation
```

## Terrain Types (12)

| Category | Terrain | Proportion | Purpose |
|---|---|---|---|
| Geometric | Ascending stairs | 10% | Stair climbing |
| Geometric | Descending stairs | 10% | Stair descent |
| Geometric | Random boxes | 10% | Rubble/boulder proxy |
| Geometric | Stepping stones | 5% | Precise foot placement |
| Geometric | Gaps | 5% | Stride gaps |
| Surface | Random rough | 10% | General uneven ground |
| Surface | Uphill slopes | 7.5% | Incline traversal |
| Surface | Downhill slopes | 7.5% | Decline traversal |
| Surface | Wave terrain | 5% | Undulating ground |
| Surface | Flat | 5% | Baseline + friction challenge |
| Compound | HF stairs (noisy) | 10% | Stairs with debris |
| Compound | Discrete obstacles | 5% | Scattered large objects |
| Compound | Rails | 5% | Balance challenge |
| Compound | Repeated boxes | 5% | Regular obstacle patterns |

## New Reward Terms

1. **velocity_modulation** (+2.0): Accept slower speeds on hard terrain
2. **body_height_tracking** (-2.0): Maintain consistent standing height
3. **contact_force_smoothness** (-0.5): Gentle foot placement
4. **stumble_penalty** (-2.0): Penalize tripping on obstacles

## Success Criteria (Post-Training Evaluation)

Using the existing `4_env_test` evaluation pipeline:
- **Friction**: Fall rate < 30% (was 70%), progress > 35m (was 27m)
- **Grass**: Progress > 30m, outperform flat policy
- **Boulder**: Progress > 20m (was 13m), reach zone 3+ consistently
- **Stairs**: Fall rate < 20% (was 15%), velocity > 0.4 m/s (was 0.21 m/s)
