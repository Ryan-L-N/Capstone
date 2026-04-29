# ANYmal-C Rough Terrain — 1-Hour Training Run

## Overview
Extended training run on the H-100 server to produce a well-trained locomotion policy.

| Parameter | Value |
|---|---|
| **Task** | Isaac-Velocity-Rough-Anymal-C-v0 |
| **RL Framework** | RSL-RL (PPO) |
| **Environments** | 1,024 |
| **Max Iterations** | 1,400 |
| **Total Timesteps** | 34,406,400 |
| **Training Time** | 3,601.71 seconds (~60 min) |
| **Computation Speed** | ~9,800 steps/s (steady state) |
| **Date** | February 12, 2026 |

## Hardware
- **GPU**: NVIDIA H100 NVL (95,830 MiB) — used ~6 GB (6.2%)
- **CPU**: Intel Xeon Platinum 8581V (60 cores / 120 threads)
- **RAM**: 1 TiB
- **Driver**: 580.126.16 | CUDA 13.0

## Software
- **Isaac Sim**: 5.1.0
- **Isaac Lab**: 0.54.3
- **PyTorch**: 2.7.0+cu126
- **Python**: 3.11.14 (conda env: env_isaaclab)

## Training Progress

| Iteration | Mean Reward | Terrain Level | Steps/s | Notes |
|---|---|---|---|---|
| 0 | -0.46 | 3.55 | 5,255 | Random policy |
| 99 | 1.95 | 0.00 | 9,833 | Learning basic gait |
| 199 | 4.83 | 0.28 | 9,849 | Steady locomotion |
| 499 | 8.67 | 2.12 | 9,922 | Climbing curriculum |
| 699 | 9.45 | 2.89 | 9,737 | Rougher terrain |
| 999 | 10.18 | 3.75 | 9,867 | Advanced terrain |
| 1399 | **10.76** | **4.71** | 9,603 | Final policy |

**Reward: -0.46 → 10.76** over 1 hour. The robot progressed through 4.7 terrain difficulty levels.

## Saved Checkpoints (on H100)
29 checkpoints saved every 50 iterations:
`model_0.pt`, `model_50.pt`, ... `model_1350.pt`, `model_1399.pt`

Located at: `/home/t2user/IsaacLab/logs/rsl_rl/anymal_c_rough/2026-02-12_20-53-32/`

## Files (This Directory)
- `model_1399.pt` — Final trained policy (iteration 1399)
- `env.yaml` — Environment configuration
- `agent.yaml` — Agent/PPO configuration
- `training_log.txt` — Full training output log (1400 iterations)

## Command Used
```bash
conda activate env_isaaclab
export OMNI_KIT_ACCEPT_EULA=YES
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Velocity-Rough-Anymal-C-v0 \
    --headless \
    --max_iterations 1400 \
    --num_envs 1024
```

## Comparison to 100-Iteration Run (Previous)

| Metric | 100 Iters (~5 min) | 1400 Iters (~1 hr) |
|---|---|---|
| Mean Reward | 1.95 | **10.76** (5.5x better) |
| Terrain Level | 0.0 | **4.71** |
| Total Timesteps | 2.45M | **34.4M** (14x more) |
| Policy Quality | Basic walking | Rough terrain traversal |
