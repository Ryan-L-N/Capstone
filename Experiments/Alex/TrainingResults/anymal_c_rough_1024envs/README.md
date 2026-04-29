# ANYmal-C Rough Terrain Training Results

## Overview
Test training run on the H-100 server (ai2ct2) to verify the full Isaac Sim + Isaac Lab stack.

| Parameter | Value |
|---|---|
| **Task** | Isaac-Velocity-Rough-Anymal-C-v0 |
| **RL Framework** | RSL-RL (PPO) |
| **Environments** | 1024 |
| **Max Iterations** | 100 |
| **Total Timesteps** | 2,457,600 |
| **Training Time** | 269.61 seconds (~4.5 min) |
| **Computation Speed** | ~9,459 steps/s |
| **Date** | February 12, 2026 |

## Hardware
- **GPU**: NVIDIA H100 NVL (95,830 MiB)
- **CPU**: Intel Xeon Platinum 8581V (60 cores / 120 threads)
- **RAM**: 1 TiB
- **Driver**: 580.126.16
- **CUDA**: 13.0

## Software
- **Isaac Sim**: 5.1.0
- **Isaac Lab**: 0.54.3
- **PyTorch**: 2.7.0+cu126
- **Python**: 3.11.14 (conda env: env_isaaclab)

## Training Progress
| Iteration | Mean Reward | Mean Episode Length | Steps/s |
|---|---|---|---|
| 1 | -0.46 | 10.95 | — |
| 25 | ~ -4.5 | ~90 | ~9,400 |
| 50 | ~ -7.0 | ~200 | ~9,400 |
| 99 | **1.95** | **260.09** | **9,459** |

The model learned to locomote on rough terrain, with reward improving from -0.46 to +1.95 over 100 iterations.

## Files
- `model_99.pt` — Final trained policy (iteration 99)
- `env.yaml` — Environment configuration
- `agent.yaml` — Agent/PPO configuration
- `training_log.txt` — Full training output log

## Command Used
```bash
conda activate env_isaaclab
export OMNI_KIT_ACCEPT_EULA=YES
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Velocity-Rough-Anymal-C-v0 \
    --headless \
    --max_iterations 100 \
    --num_envs 1024
```

## Notes
- First run on a fresh install requires 15-30 min of extension caching and Warp kernel compilation
- Subsequent runs start training within ~30 seconds
- The `libglu1-mesa` package must be installed (`sudo apt install libglu1-mesa`)
- Use `screen` sessions for training to allow safe monitoring
