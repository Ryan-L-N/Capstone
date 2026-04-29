# Attempt 5: Train From Scratch with 235-dim Obs + Terrain Curriculum

## Why Attempt 5?

Attempts 1–4 all tried to **fine-tune** the pre-trained 48hr flat-terrain policy by expanding it from 48-dim to 235-dim observations and training on rough terrain. Every attempt collapsed at the freeze/unfreeze boundary:

| Attempt | Failure | Root Cause |
|---------|---------|-----------|
| #1 | Immediate collapse | Loaded full checkpoint — critic trained on 14 rewards, env has 19 |
| #2 | Noise explosion (0.65 → 5.75 in 247 iters) | Std left trainable during actor freeze — entropy bonus inflated it |
| #3 | Catastrophic forgetting at unfreeze (ep_len 206 → 2.5 in 30 iters) | PPO hyperparams too aggressive for fine-tuning (LR 1e-4, clip 0.2) |
| #4 | Gradual collapse (ep_len 150 → 4 over 100 iters post-unfreeze) | Even ultra-conservative PPO (LR 1e-5, clip 0.1) couldn't prevent critic-staleness cascade |

**The fundamental insight:** The critic-first warmup approach doesn't work. The critic learns value estimates for the *frozen* actor, but the moment the actor changes — even at microscopic learning rates — the values become stale, advantages go wrong, and it cascades into collapse. The fine-tuning surgery is the problem.

## The Attempt 5 Approach

**Train a fresh 235-dim policy from scratch on flat terrain, with terrain curriculum to gradually introduce harder terrains.**

- No checkpoint loading — random initialization
- No freeze/unfreeze — actor and critic grow up together from iter 0
- No LR warmup — standard adaptive KL handles everything
- Height scan dims (187) are zeros on flat terrain, become useful as terrain curriculum promotes robots to rougher ground
- Standard from-scratch PPO hyperparameters (LR 1e-3, clip 0.2, entropy 0.005)

## Terrain Curriculum

All robots start at difficulty level 0 (flat/minimal). The curriculum auto-promotes as they learn to walk.

| Terrain | Proportion | Difficulty 0 (start) | Difficulty 9 (hard) |
|---------|-----------|---------------------|---------------------|
| **Flat** | 20% | Truly flat | Truly flat |
| **Random rough (uneven)** | 20% | 0.02m noise | 0.15m noise |
| **Boxes (boulders)** | 15% | 0.05m height | 0.25m height |
| **Stairs up** | 15% | 0.05m steps | 0.25m steps |
| **Stairs down** | 15% | 0.05m steps | 0.25m steps |
| **Friction plane** | 10% | Flat + low friction | Flat + low friction |
| **Vegetation plane** | 5% | Flat + drag | Flat + drag |

Grid: 10 rows (difficulty) × 30 cols (variety) = 300 patches, 8m × 8m each.

At difficulty 0, even non-flat terrains are barely perceptible (5cm steps, 2cm roughness). The robot first learns to walk on near-flat ground, then the curriculum pushes it to progressively harder terrain.

## PPO Hyperparameters

Standard from-scratch (same as original 48hr policy):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 1e-3 | Standard from-scratch |
| Clip ratio | 0.2 | Standard PPO |
| Entropy coef | 0.005 | Standard exploration bonus |
| Learning epochs | 5 | Standard |
| KL target | 0.01 | Looser — needs exploration from scratch |
| init_noise_std | 1.0 | Will converge naturally |
| Architecture | [512, 256, 128] | Same as 48hr |
| Steps per env | 24 | Standard |
| Mini-batches | 4 | Standard |
| Gamma | 0.99 | Standard |
| Lambda (GAE) | 0.95 | Standard |

## Training Configuration

- **Envs:** 16,384 (H100)
- **Max iterations:** 15,000 (~48hrs estimated)
- **Progressive DR:** Friction, push, force, mass expand over 15K iters
- **Episode length:** 30s
- **Observations:** 235-dim (48 proprio + 187 height scan)
- **Rewards:** 19 terms (14 base + 5 custom)
- **Terrain curriculum:** `max_init_terrain_level=0` — all robots start flat

## Progressive Domain Randomization

Same schedule as Attempts 1–4 — proven to prevent "max DR from iter 0" failure:

| Parameter | Start (iter 0) | End (iter 15K) |
|-----------|---------------|----------------|
| Static friction | [0.3, 1.3] | [0.1, 1.5] |
| Dynamic friction | [0.25, 1.1] | [0.08, 1.2] |
| Push velocity | ±0.5 m/s | ±1.0 m/s |
| External force | ±3.0 N | ±6.0 N |
| External torque | ±1.0 N·m | ±2.5 N·m |
| Mass offset | ±5.0 kg | ±7.0 kg |

## Files Created

| File | Purpose |
|------|---------|
| `configs/scratch_terrain_cfg.py` | Terrain curriculum (7 types, flat start) |
| `configs/scratch_ppo_cfg.py` | Standard from-scratch PPO config |
| `configs/scratch_env_cfg.py` | Env config (reuses rewards/obs from finetune, swaps terrain) |
| `train_from_scratch.py` | Clean training script (no freeze/unfreeze/checkpoint) |
| `scripts/train_scratch_h100.sh` | H100 production launch |
| `scripts/train_scratch_local_debug.sh` | Local debug (64 envs, 10 iters) |

## Expected Training Trajectory

1. **Iters 0–500:** Robot learns to stand, episode length climbs from ~2 to ~20
2. **Iters 500–2000:** Robot learns basic trot gait, episode length → 100+
3. **Iters 2000–5000:** Curriculum starts promoting to level 1–3 terrain, noise std converges toward 0.6–0.7
4. **Iters 5000–10000:** Robot handles moderate terrain, terrain levels → 4–6
5. **Iters 10000–15000:** DR fully expanded, robot handles hard terrain, terrain levels → 7–9

## What Success Looks Like

- Episode length >500 steps (10+ seconds) on terrain level 5+
- Terrain levels steadily climbing (not stuck at 0 or collapsing)
- Noise std converges to 0.5–0.8 range naturally
- Body contact termination rate <30%
- No sudden collapses (the whole point of avoiding fine-tuning)

## Tradeoff

We spend ~48hrs re-learning basic locomotion that the 48hr policy already knew. But we completely skip the fragile fine-tuning surgery that failed 4 times. The terrain curriculum and progressive DR ensure a smooth difficulty ramp rather than the abrupt freeze/unfreeze transition.
