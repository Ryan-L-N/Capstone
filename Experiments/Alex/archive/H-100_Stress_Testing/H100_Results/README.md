# H-100 Stress Test Results

## Date: February 12, 2026

## Test Configuration
- **Task**: Isaac-Velocity-Rough-Anymal-C-v0 (ANYmal-C rough terrain locomotion)
- **RL Framework**: RSL-RL (PPO)
- **Iterations per test**: 10
- **Mode**: Headless
- **Env counts tested**: 1,024 → 65,536 (7 tests, doubling each time)

## Hardware
- **GPU**: NVIDIA H100 NVL (95,830 MiB / ~96 GB VRAM)
- **CPU**: Intel Xeon Platinum 8581V (60 cores / 120 threads)
- **RAM**: 1 TiB DDR5
- **Driver**: 580.126.16 | CUDA 13.0

---

## Results Summary

| Envs | Steps/s | Iter Time (s) | Total Time (s) | Peak VRAM (MiB) | GPU Util (%) | Temp (°C) | Power (W) | Peak RAM (MiB) | OOM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1,024 | 8,965 | 2.74 | 30.88 | 5,960 | 36% | 40 | 110 | 5,912 | NO |
| 2,048 | 15,662 | 3.14 | 35.17 | 6,522 | 45% | 42 | 126 | 6,212 | NO |
| 4,096 | 24,496 | 4.01 | 44.56 | 7,882 | 57% | 45 | 146 | 6,904 | NO |
| 8,192 | 36,447 | 5.39 | 59.03 | 10,240 | 65% | 49 | 171 | 8,367 | NO |
| 16,384 | 37,865 | 10.38 | 101.15 | 13,442 | 79% | 57 | 204 | 11,042 | NO |
| 32,768 | 40,944 | 19.21 | 183.72 | 20,950 | 100% | 63 | 250 | 17,011 | NO |
| 65,536 | **69,090** | 22.77 | 227.82 | 35,344 | 100% | 65 | 299 | 27,890 | NO |

---

## Key Findings

### 1. No OOM — Even at 65,536 Environments
The H100's 96 GB VRAM comfortably handled all env counts. Peak VRAM at 65K envs was only **35.3 GB (37% of capacity)**. The limiting factor is not VRAM.

### 2. Throughput Sweet Spot: 8,192–16,384 Envs (Efficiency) / 65,536 (Raw Speed)
- **Linear scaling** from 1K to 8K envs (steps/s nearly proportional to env count)
- **Plateau** at 8K–16K (37,865 vs 36,447 — only 4% gain for 2x envs)
- **Second wind at 32K–65K** — throughput jumps to 69K steps/s at 65K envs
- At 65K, per-iteration time is 22.77s but each iteration processes 65,536 × 24 = 1.57M steps

### 3. GPU Far From Thermal Limits
- Peak temperature: **65°C** (at 65K envs) — well below throttle point (~83°C)
- Peak power: **299W** out of 400W TDP (75% power budget)
- No thermal throttling observed at any env count

### 4. RAM Scales Linearly
- 1K envs: ~6 GB RAM → 65K envs: ~28 GB RAM
- The 1 TiB of system RAM is far from a bottleneck

### 5. PhysX Collision Stack Overflow at 65K Envs
At 65,536 envs, PhysX reported `collisionStackSize buffer overflow` — contacts were dropped during simulation. This means the 65K results have **degraded physics fidelity**. For production training, 32K envs is the practical maximum without physics errors.

---

## Scaling Analysis

### Throughput Scaling Factor (vs 1,024 envs baseline)
| Envs | Speedup vs 1K | Env Count Multiplier | Efficiency |
|---:|---:|---:|---:|
| 1,024 | 1.00x | 1x | 100% |
| 2,048 | 1.75x | 2x | 87% |
| 4,096 | 2.73x | 4x | 68% |
| 8,192 | 4.07x | 8x | 51% |
| 16,384 | 4.22x | 16x | 26% |
| 32,768 | 4.57x | 32x | 14% |
| 65,536 | 7.71x | 64x | 12% |

Efficiency drops as env count grows (expected), but raw throughput keeps climbing.

### VRAM Scaling
- ~5 GB base overhead
- ~0.46 MiB per additional environment
- Projected OOM point: ~196K envs (theoretical, physics errors would occur first)

---

## Monitoring Data (Per-Test Directories)
Each `envs_XXXX/` directory contains:
- `training.log` — Full training output with per-iteration metrics
- `gpu_metrics.csv` — GPU util/VRAM/temp/power sampled every 1 second
- `ram_metrics.csv` — System RAM usage sampled every 1 second
- `system_info.txt` — System state before and after test
- `summary.txt` — Extracted peak values

Note: CPU metrics (`mpstat`) were not collected due to missing `sysstat` package on the server.

---

## Recommendations for Production Training

| Scenario | Recommended Envs | Reason |
|---|---:|---|
| **Quick iteration / debugging** | 1,024 | Fast startup, low resource use |
| **Balanced training** | 4,096 – 8,192 | Best efficiency (steps/s per env) |
| **Maximum throughput** | 32,768 | Highest safe throughput without physics errors |
| **Research / maximum speed** | 65,536 | Highest steps/s but degraded physics |
