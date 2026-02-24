# Hybrid Student-Teacher Reinforcement Learning (hybrid_ST_RL)

## Overview

A two-stage training approach for robust quadruped locomotion on the Boston Dynamics Spot robot, combining **progressive fine-tuning** with **teacher-student distillation** inspired by CMU's Extreme Parkour research.

**Problem:** A 100-hour from-scratch training run on NVIDIA H100 NVL 96GB failed after 10K iterations (17.6 billion timesteps) with zero learning progress. The policy never learned to stand for more than 7 seconds across 12 terrain types with aggressive domain randomization.

**Solution:** Instead of training from scratch, fine-tune the existing 48-hour rough terrain policy with a progressive difficulty curriculum and domain randomization schedule. Optionally apply teacher-student distillation for further improvement on the hardest terrains.

---

## Training Pipeline

### Stage 1: Progressive Fine-Tuning (~69 hours on H100)

Initialize from the working 48hr rough policy checkpoint (`model_27500.pt`, architecture `[512, 256, 128]`) and fine-tune on 12 terrain types. The key innovation is **progressive domain randomization**: DR parameters start near the 48hr training conditions and linearly expand to harder ranges over 15,000 iterations.

This avoids the failure mode of the previous attempt, which applied maximum difficulty (friction as low as 0.05, push velocities of 1.5 m/s) from iteration 0 to a randomly initialized network.

| Parameter | Start (iter 0) | End (iter 15K+) |
|---|---|---|
| Static friction | [0.3, 1.3] | [0.1, 1.5] |
| Push velocity | +/-0.5 m/s | +/-1.0 m/s |
| External force | +/-3.0 N | +/-6.0 N |
| Mass offset | +/-5.0 kg | +/-7.0 kg |

### Stage 2: Teacher-Student Distillation (Optional, ~36 hours)

If Stage 1 alone doesn't achieve target metrics, apply the teacher-student approach from Extreme Parkour:

**Phase 2a - Teacher (24hr):** Add privileged observations (exact friction, terrain type, clean contact forces) to the Stage 1 policy. The teacher learns to leverage this extra information for better terrain adaptation.

**Phase 2b - Student (12hr):** Distill the teacher's behavior back into a standard-observation policy using a combined PPO + behavior cloning loss. The student learns terrain-adaptive behavior from height scan alone, without privileged information.

---

## Architecture

```
Policy Network (Actor & Critic):
  Input:  235 dims (48 proprioceptive + 187 height scan)
  Hidden: [512, 256, 128] with ELU activation
  Output: 12 dims (joint position targets)

Teacher Network (Stage 2 only):
  Input:  254 dims (235 standard + 19 privileged)
  Hidden: [512, 256, 128] with ELU activation
  Output: 12 dims
```

---

## Target Evaluation Environments

Performance is measured on 4 evaluation environments from `4_env_test/`:

| Environment | Challenge | Current Rough Policy | Target |
|---|---|---|---|
| **Friction** | Decreasing friction (1.0 to 0.2) | 69% fall rate, 26.1m | < 40% falls, > 35m |
| **Grass** | Vegetation drag (0 to 20 N*s/m) | 19% fall rate, 24.1m | < 15% falls, > 28m |
| **Boulder** | Random rough + boxes (increasing) | 70% fall rate, 13.4m | < 50% falls, > 20m |
| **Stairs** | Step height (5cm to 25cm) | 14% fall rate, 11.3m | < 12% falls, > 15m |

---

## Literature & References

### Primary Inspiration

**1. Extreme Parkour with Legged Robots**
- Authors: Xuxin Cheng*, Kexin Shi*, Ananye Agarwal, Deepak Pathak (Carnegie Mellon University)
- Published: ICRA 2024 (arXiv: 2309.14341)
- URL: https://extreme-parkour.github.io/
- GitHub: https://github.com/chengxuxin/extreme-parkour

Key contributions applied to this project:
- **Two-phase teacher-student training:** Train a teacher with privileged terrain information (scandots), then distill to a student that operates from onboard sensors only. We adapt this for height scan (teacher gets exact friction/terrain type, student gets noisy height scan).
- **Universal reward design via inner products:** Single reward function works across all terrain types without per-terrain reward engineering. The heading alignment reward `r = min(<v, d_w>, v_cmd)` naturally adapts to different obstacles.
- **Automatic terrain curriculum:** Robots are promoted to harder terrain when they traverse >50% of expected distance, and demoted when they fail. This self-regulating curriculum finds the learning frontier automatically.
- **Training efficiency:** Achieved parkour-capable policies in <20 hours on a single RTX 3090, demonstrating that careful initialization and curriculum design matter more than raw compute.

### Supporting Literature

**2. Closing the Sim-to-Real Gap: Training Spot Quadruped Locomotion with NVIDIA Isaac Lab**
- Authors: NVIDIA Isaac Lab Team
- URL: https://developer.nvidia.com/blog/closing-the-sim-to-real-gap-training-spot-quadruped-locomotion-with-nvidia-isaac-lab/
- Relevance: Baseline Spot locomotion training in Isaac Lab. Our 48hr rough policy follows this configuration (6 terrain types, `ROUGH_TERRAINS_CFG`, PPO with RSL-RL). Provides the checkpoint we fine-tune from.

**3. ANYmal Parkour: Learning Agile Navigation for Quadrupedal Robots**
- Authors: David Hoeller et al. (ETH Zurich / NVIDIA)
- Published: Science Robotics, 2024 (DOI: 10.1126/scirobotics.adi7566)
- Relevance: Teacher-student paradigm for terrain curriculum. Demonstrates that privileged information (exact terrain geometry) enables faster teacher learning, and distillation to proprioception-only students preserves most performance. Validates the two-phase approach.

**4. Parkour in the Wild: Unified Agile Locomotion**
- Authors: ETH Zurich / NVIDIA
- Published: arXiv: 2505.11164, 2025
- Relevance: Shows that a single policy can handle diverse parkour skills (jumping, climbing, crawling) through unified training with terrain curriculum. Supports our goal of one policy for 12 terrain types.

**5. CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion**
- Published: 2024
- URL: https://www.researchgate.net/publication/383934038
- Relevance: Explores concurrent (rather than sequential) teacher-student training. The teacher and student train simultaneously, with the student learning from both RL rewards and teacher actions. An alternative to our sequential Stage 1 -> Stage 2 approach.

**6. Scaling Rough Terrain Locomotion with Automatic Curriculum RL**
- Published: arXiv: 2601.17428, 2026
- Relevance: Demonstrates that automatic curriculum (promotion/demotion based on performance) is critical for scaling to many terrain types. Directly informs our terrain curriculum implementation using `mdp.terrain_levels_vel`.

### Isaac Lab & RSL-RL Framework

**7. Isaac Lab Documentation**
- Terrain API: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.terrains.html
- RL Overview: https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html
- Performance Benchmarks: https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/performance_benchmarks.html
- Migration from Isaac Gym: https://isaac-sim.github.io/IsaacLab/main/source/migration/comparing_simulation_isaacgym.html

**8. RSL-RL (Robotic Systems Lab RL)**
- GitHub: https://github.com/leggedrobotics/rsl_rl
- Used for PPO training with `OnPolicyRunner`. Checkpoint loading via `runner.load()` requires exact architecture match (`load_state_dict` with `strict=True`).

---

## Why the Previous Training Failed

### 100-Hour From-Scratch Run (Failed)

| Issue | Detail |
|---|---|
| **Architecture mismatch** | Used [1024,512,256] (4x larger) instead of [512,256,128]. Could not load working checkpoint due to shape mismatch in `load_state_dict()`. |
| **No initialization** | Trained from random weights. The policy had to simultaneously learn to stand, balance, walk, AND navigate 12 terrain types with extreme DR. |
| **Maximum DR from iteration 0** | Friction as low as 0.05 (oil on steel), push velocity 1.5 m/s, external force 8N. Even a policy that stumbles into standing immediately gets knocked down. |
| **Contradictory gradients** | Robots on ice (friction 0.05) need slow, careful movements. Robots on sandpaper (friction 1.5) benefit from aggressive gaits. Same gradient update tries to optimize for both simultaneously. |
| **No progressive curriculum for DR** | Terrain had curriculum (via `terrain_levels_vel`), but DR was fixed at maximum. The terrain curriculum couldn't advance because the policy kept falling due to extreme DR, not terrain difficulty. |

### This Approach Fixes All Five Issues

1. **Same architecture** [512,256,128] -- checkpoint loads directly
2. **Warm start** from model_27500.pt -- policy already walks
3. **Progressive DR** -- starts at 48hr-like values, expands over 15K iterations
4. **Coherent gradients** -- early training has narrow friction range, so gradients are consistent
5. **DR and terrain curriculum aligned** -- DR expands as the policy gets stronger, not before

---

## Environment Count & Iteration Scaling — Why 16,384 x 25,000

Choosing the right number of parallel environments and training iterations is a
core design decision. More environments does **not** slow training down — it
makes each gradient update higher quality because the network sees a bigger,
more diverse batch of experience.

### How GPU parallelism works

Isaac Lab runs physics on the GPU itself (GPU PhysX). Simulating 16K robots
costs only ~2x the wall-clock time of 8K because the GPU cores work in
parallel. But you get 2x the data per iteration, so each gradient update is
more accurate and stable. The network converges in fewer iterations.

### The tradeoff table

| Envs | Steps / iter | ~Time / iter | Robots per terrain (12 types) | Total steps @ iters | Wall time |
|------|-------------|-------------|-------------------------------|---------------------|-----------|
| 8,192 | 196K | ~6 s | 683 | 5.9B @ 30K | ~50 h |
| **16,384** | **393K** | **~10 s** | **1,365** | **9.8B @ 25K** | **~69 h** |
| 32,768 | 786K | ~18 s | 2,730 | 11.8B @ 15K | ~75 h |
| 65,536 | 1.57M | ~40 s | 5,461 | 15.7B @ 10K | ~111 h |

### Why 16,384 is the sweet spot

1. **Terrain coverage:** 1,365 robots per terrain type — every terrain gets
   well-sampled each iteration. With 8K we only had 683 per type, which is thin
   for 12 terrains.
2. **Total experience:** 9.8 billion steps — almost 2x what the successful 48hr
   run needed (5.3B), which accounts for doubling the terrain types from 6 to 12
   while benefiting from the warm start.
3. **Wall time:** ~10 s/iter x 25K = ~69 hours. Fits the 72-hour budget with
   margin for checkpointing overhead.
4. **DR schedule alignment:** Progressive DR expands over 15K iterations (60% of
   training), then 10K iterations at full difficulty. Good ratio.
5. **VRAM headroom:** The [512,256,128] network is half the size of the failed
   100hr run's [1024,512,256]. Running 16K envs uses ~30-40 GB on the 96 GB H100.
6. **Diminishing returns beyond 16K:** Going 8K->16K doubles per-terrain
   coverage (683->1,365) — meaningful. Going 16K->32K (1,365->2,730) helps
   less but costs 80% more time per iteration.

### Why not 65,536 (what we ran last time)?

The failed 100hr run used 65,536 envs with a [1024,512,256] network. Each
iteration took ~43 seconds. That was overkill — the extra robots-per-terrain
beyond ~1,400 gave diminishing returns on gradient quality, while ballooning
wall-clock time. The failure was due to training-from-scratch + wrong
architecture, not insufficient parallelism.

### Comparison to the successful 48hr run

The 48hr rough policy trained from scratch with 8,192 envs x 27,500 iterations
on 6 terrain types = 5.3B total steps. We are fine-tuning (not from scratch) on
12 terrain types (2x). The 16,384 x 25,000 config gives 9.8B steps — roughly
1.9x the sample budget, which accounts for the harder task while benefiting from
the warm start.

---

## Hardware

| Component | Specification |
|---|---|
| **Training GPU** | NVIDIA H100 NVL 96GB HBM3 |
| **Training server** | ai2ct2 (172.24.254.24) |
| **Training envs** | 16,384 parallel (Stage 1) |
| **Training iterations** | 25,000 (Stage 1) |
| **Local testing GPU** | NVIDIA RTX 2000 Ada 8GB |
| **Simulator** | Isaac Sim 5.1.0.0, Isaac Lab 0.54.2 |
| **RL Framework** | RSL-RL (PPO), PyTorch 2.7.0+cu128 |
| **Python** | 3.11 (conda env: isaaclab311 / env_isaaclab) |

---

## Project Context

AI2C Tech Capstone -- MS for Autonomy, Carnegie Mellon University, February 2026.

Team: Alex Santiago, Ryan L-N, Colby (+ collaborators)

Repository: `github.com/Ryan-L-N/Capstone.git` (branch: `development`)
