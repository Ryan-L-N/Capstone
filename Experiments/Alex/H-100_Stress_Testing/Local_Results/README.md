# Local Stress Test Results — RTX 2000 Ada Generation

## Date: February 12, 2026

## Hardware
- **GPU**: NVIDIA RTX 2000 Ada Generation (8,188 MiB / ~8 GB VRAM, 35W TDP)
- **CPU**: Intel Core Ultra 9 185H (16 cores / 22 threads)
- **RAM**: 32 GB DDR5
- **Driver**: 573.44 | CUDA 12.8
- **Isaac Sim**: 5.1.0 | **Isaac Lab**: 0.54.2

## Results

| Envs | Steps/s | Iter Time (s) | Total Time (s) | Peak VRAM (MiB) | GPU Util (%) | Temp (°C) | Power (W) | OOM |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1,024 | 8,463 | 2.90 | 31.72 | 3,643 | 68% | 58 | 29.5 | NO |
| 2,048 | 10,036 | 4.90 | 48.14 | 4,161 | 84% | 64 | 29.9 | NO |
| 4,096 | 13,239 | 7.43 | 85.11 | 5,465 | 93% | 68 | 38.4 | NO |

## Key Findings
- No OOM even at 4,096 envs (5.5 GB of 8 GB used — 67%)
- GPU approaching saturation at 4,096 envs (93% util)
- 8,192 envs would likely OOM (estimated ~7.8+ GB needed based on scaling)
- Power exceeded 35W TDP at 4,096 envs (38.4W — GPU boost behavior)

## H100 Comparison (at matching env counts)

| Envs | Local Steps/s | H100 Steps/s | H100 Speedup |
|---:|---:|---:|---:|
| 1,024 | 8,463 | 8,965 | **1.06x** |
| 2,048 | 10,036 | 15,662 | **1.56x** |
| 4,096 | 13,239 | 24,496 | **1.85x** |

The H100 advantage grows with env count. At 4K envs, the H100 is nearly 2x faster. Beyond 4K the local GPU can't compete — the H100 scales to 65K envs and 69K steps/s (8.2x local peak).
