# How Fast Can a Robot Learn to Walk? — Stress-Testing an H100 GPU for Sim-to-Real Locomotion Training

> *"We gave NVIDIA's most powerful GPU 65,536 virtual robots and said 'teach them to walk.' Here's what happened."*

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Why This Project Exists](#2-why-this-project-exists)
3. [The Hardware: David vs. Goliath](#3-the-hardware-david-vs-goliath)
4. [Technical Architecture: How All the Pieces Fit Together](#4-technical-architecture-how-all-the-pieces-fit-together)
5. [The Codebase: A Map of the Territory](#5-the-codebase-a-map-of-the-territory)
6. [Technology Stack and Why We Chose It](#6-technology-stack-and-why-we-chose-it)
7. [Results: The Numbers Tell a Story](#7-results-the-numbers-tell-a-story)
8. [The Bug Journal: What Broke and How We Fixed It](#8-the-bug-journal-what-broke-and-how-we-fixed-it)
9. [Lessons for Future Engineers](#9-lessons-for-future-engineers)
10. [Recommendations for Maximizing Training Runs](#10-recommendations-for-maximizing-training-runs)
11. [Where We'll Hit Walls Next](#11-where-well-hit-walls-next)
12. [Final Thoughts](#12-final-thoughts)

---

## 1. The Big Picture

Imagine you're teaching a dog to walk across a rocky hiking trail. You could let one dog practice for a year, falling and getting up over and over. Or you could clone that dog 65,536 times, put each clone on a slightly different trail, and let them *all* practice simultaneously — sharing what they learn after every attempt. In one hour, you'd have the equivalent of 7.5 years of hiking experience.

That's essentially what we did — except the "dog" is a simulated quadruped robot called ANYmal-C, the "trails" are randomly generated rough terrain inside NVIDIA's Isaac Sim physics engine, and the "clones" are parallel simulation environments running on GPU hardware.

**The core question**: *How many parallel environments can we run before the hardware buckles, and what's the optimal number for training the best locomotion policy in the shortest time?*

We tested two machines — an NVIDIA H100 data center GPU with 96 GB of VRAM (the kind of hardware that costs as much as a luxury car), and an NVIDIA RTX 2000 Ada laptop GPU with 8 GB of VRAM (the kind you'd find in a mid-range engineering workstation). We pushed both to their limits.

---

## 2. Why This Project Exists

In reinforcement learning (RL), the single biggest bottleneck is **sample efficiency** — how many experiences the agent needs before it learns a good policy. Modern RL algorithms like PPO (Proximal Policy Optimization) are hungry. They need *millions* of timesteps to converge.

The breakthrough insight of GPU-accelerated simulation is that you can generate those timesteps in parallel. Instead of running one robot and waiting for it to stumble through 34 million steps sequentially, you run thousands of robots simultaneously. The physics, the rendering, the neural network forward passes, the gradient updates — it all happens on the GPU.

But there's a catch. More parallel environments means more memory, more compute, and more heat. At some point, you either run out of VRAM (the GPU equivalent of RAM), the physics engine starts dropping calculations, or the GPU throttles itself to avoid melting.

**We needed to find that point.** And we needed to find the sweet spot *before* that point — the number of environments that maximizes training throughput without sacrificing physics accuracy or hardware longevity.

---

## 3. The Hardware: David vs. Goliath

Here's what we were working with. Picture a Ferrari F1 car versus a Honda Civic:

| Spec | H100 NVL ("The Ferrari") | RTX 2000 Ada ("The Civic") | Ratio |
|---|---|---|---|
| **GPU Memory (VRAM)** | 96 GB | 8 GB | **12x** |
| **Power Budget (TDP)** | 400W | 35W | **11.4x** |
| **CPU** | Xeon 8581V (60 cores) | Core Ultra 9 185H (16 cores) | **3.75x** |
| **System RAM** | 1 TiB (1,024 GB) | 32 GB | **32x** |
| **Cost (approximate)** | ~$30,000+ (GPU alone) | ~$2,000 (whole laptop) | **15x+** |

The H100 is NVIDIA's flagship data center GPU — the same chip powering ChatGPT, Claude, and most large-scale AI training in the world right now. Our specific variant (NVL) has 96 GB of HBM3 memory, which is monstrously large for a single GPU.

The RTX 2000 Ada is a professional mobile GPU. Solid for CAD work, light ML training, and running inference — but it's a laptop chip designed to sip power, not chug it.

The question isn't *whether* the H100 is faster — of course it is. The question is: **how much faster, and at what scale does the gap actually matter?**

Spoiler: the answer surprised us.

---

## 4. Technical Architecture: How All the Pieces Fit Together

Here's the full pipeline, from SSH command to trained neural network:

```
┌─────────────────────────────────────────────────────────────────────┐
│  YOUR LAPTOP (Windows 11)                                           │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐    │
│  │ SSH / SCP     │────▶│ H100 Server  │     │ Local Training   │    │
│  │ (paramiko)    │◀────│ (Ubuntu)     │     │ (RTX 2000 Ada)   │    │
│  └──────────────┘     └──────┬───────┘     └────────┬─────────┘    │
│                              │                       │              │
│                    ┌─────────▼─────────┐   ┌────────▼────────┐     │
│                    │ stress_test.sh     │   │local_stress_    │     │
│                    │ (bash on server)   │   │test.py (Python) │     │
│                    └─────────┬─────────┘   └────────┬────────┘     │
│                              │                       │              │
│                    ┌─────────▼─────────────────────────────────┐    │
│                    │           Isaac Lab + Isaac Sim            │    │
│                    │    ┌──────────────────────────────┐       │    │
│                    │    │  RSL-RL (PPO Algorithm)       │       │    │
│                    │    │  ┌────────────────────────┐   │       │    │
│                    │    │  │ Actor-Critic Network    │   │       │    │
│                    │    │  │ (PyTorch on CUDA)       │   │       │    │
│                    │    │  └────────────────────────┘   │       │    │
│                    │    └──────────────────────────────┘       │    │
│                    │    ┌──────────────────────────────┐       │    │
│                    │    │  Physics (PhysX on GPU)       │       │    │
│                    │    │  ┌────────┐ ┌────────┐       │       │    │
│                    │    │  │ Env #1 │ │ Env #2 │ ...   │       │    │
│                    │    │  │ANYmal-C│ │ANYmal-C│       │       │    │
│                    │    │  │+terrain│ │+terrain│       │       │    │
│                    │    │  └────────┘ └────────┘       │       │    │
│                    │    │  (×1,024 to ×65,536)         │       │    │
│                    │    └──────────────────────────────┘       │    │
│                    └───────────────────────────────────────────┘    │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ MONITORING LAYER (runs in parallel during training)          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │   │
│  │  │nvidia-smi│  │ mpstat   │  │  free -m  │  │ training   │  │   │
│  │  │ GPU stats│  │ CPU stats│  │ RAM stats │  │ log parser │  │   │
│  │  │ @1s      │  │ @1s      │  │ @1s       │  │            │  │   │
│  │  └────┬─────┘  └────┬─────┘  └─────┬────┘  └─────┬──────┘  │   │
│  │       ▼              ▼              ▼              ▼         │   │
│  │  gpu_metrics   cpu_metrics    ram_metrics    training.log    │   │
│  │  .csv          .txt           .csv                          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                    ┌─────────▼─────────┐                           │
│                    │ Results Collection │                           │
│                    │ combined_results   │                           │
│                    │ .csv + README.md   │                           │
│                    └───────────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘
```

### The Data Flow in Plain English

1. **You sit at your laptop** and SSH into the H100 server (a beefy Ubuntu machine sitting somewhere on the network at `172.24.254.24`).

2. **You launch a bash script** (`stress_test.sh`) that loops through environment counts: 1,024 → 2,048 → 4,096 → 8,192 → 16,384 → 32,768 → 65,536.

3. **For each env count**, the script:
   - Spins up background monitoring processes (GPU metrics, CPU metrics, RAM usage — all sampled every second)
   - Launches Isaac Lab's training script, which boots up Isaac Sim's physics engine
   - Waits for 10 training iterations to complete
   - Kills the monitors and extracts peak metrics

4. **Isaac Sim creates N virtual worlds** — each one containing an ANYmal-C quadruped robot standing on procedurally generated rough terrain. All N worlds are simulated simultaneously on the GPU using PhysX.

5. **RSL-RL's PPO algorithm** collects experience from all environments, computes a policy gradient, and updates the neural network weights. One "iteration" = collect 24 steps from every environment, then do a learning update.

6. **After all 7 tests**, the script generates a `combined_results.csv` with one row per env count, summarizing throughput, VRAM usage, temperature, power draw, and whether it OOM'd.

7. **You SCP the results** back to your laptop and analyze them.

For the local tests, the same flow runs on your laptop using a Python script (`local_stress_test.py`) instead of bash — since Windows doesn't natively run bash scripts, and we wanted the same monitoring capabilities through `nvidia-smi` subprocess calls.

---

## 5. The Codebase: A Map of the Territory

Here's how the project is organized and what each piece does:

```
Capstone/
└── Experiments/
    └── Alex/
        ├── H-100_Stress_Testing/
        │   ├── STRESS_TEST_PLAN.md          ← The battle plan (written before any code ran)
        │   ├── hardware_stress_test_summary.md  ← You are here
        │   ├── H100_Results/
        │   │   ├── combined_results.csv     ← Single CSV with all 7 test results
        │   │   ├── README.md                ← H100-specific analysis
        │   │   └── envs_XXXX/              ← Per-test directories (×7)
        │   │       ├── training.log         ← Raw training output
        │   │       ├── gpu_metrics.csv      ← Second-by-second GPU telemetry
        │   │       ├── ram_metrics.csv      ← Second-by-second RAM usage
        │   │       ├── system_info.txt      ← Before/after system snapshot
        │   │       └── summary.txt          ← Extracted peak values
        │   └── Local_Results/
        │       ├── combined_results.csv     ← Single CSV with all 3 local tests
        │       ├── README.md                ← Local-specific analysis + comparison
        │       └── envs_XXXX/              ← Per-test directories (×3)
        │           └── (same structure as above)
        └── TrainingResults/
            ├── anymal_c_rough_1024envs/     ← 100-iteration smoke test
            │   ├── model_99.pt              ← Saved policy weights
            │   ├── agent.yaml, env.yaml     ← Configs (reproducibility!)
            │   ├── training_log.txt         ← Full log
            │   └── README.md
            └── anymal_c_rough_1hour/        ← 1,400-iteration production run
                ├── model_1399.pt            ← Final trained policy
                ├── agent.yaml, env.yaml
                ├── training_log.txt
                └── README.md

# Scripts (workspace root):
stress_test.sh          ← Bash script deployed to H100 via SCP
local_stress_test.py    ← Python equivalent for Windows/local GPU
h100_run.py             ← Paramiko-based SSH command runner
paramiko_check.py       ← Quick server health check
get_ram_metrics.py      ← One-off helper for parsing RAM data from server
```

### How the Pieces Connect

Think of it like a restaurant kitchen:

- **`STRESS_TEST_PLAN.md`** is the **recipe** — written before we cooked anything. It specifies what we're testing, what metrics we'll collect, and what we expect to find.
- **`stress_test.sh`** is the **chef** — the script that actually does the work on the H100 server.
- **`local_stress_test.py`** is the **sous chef** — does the same job locally, adapted for Windows.
- **`h100_run.py`** is the **intercom** — lets us send commands to the server from our laptop via SSH.
- **The `envs_XXXX/` directories** are the **lab notebooks** — raw data from every experiment.
- **`combined_results.csv`** is the **final dish** — the clean, structured data we actually analyze.
- **The README files** are the **food critic's review** — human-readable analysis of what the data means.

---

## 6. Technology Stack and Why We Chose It

### NVIDIA Isaac Sim + Isaac Lab

**What it is**: Isaac Sim is NVIDIA's GPU-accelerated robotics simulator built on Omniverse. Isaac Lab is the RL training framework that sits on top of it.

**Why we chose it**: Isaac Sim runs the *entire* simulation pipeline on the GPU — physics, terrain generation, sensor simulation, everything. Traditional simulators (MuJoCo, Gazebo) run physics on the CPU and transfer data to the GPU for the neural network, creating a bottleneck. Isaac Sim eliminates that bottleneck. When you have 65,536 environments, the GPU-native approach isn't just faster — it's the *only* approach that's even feasible.

**The analogy**: Using a CPU-based simulator for 65K environments would be like trying to fill a swimming pool with a garden hose. Isaac Sim is the fire hydrant.

### RSL-RL (PPO)

**What it is**: The Robotic Systems Lab's implementation of Proximal Policy Optimization, a reinforcement learning algorithm.

**Why we chose it**: PPO is the workhorse of modern robot RL. It's stable (doesn't diverge easily), parallelizable (benefits directly from more environments), and battle-tested (used by Boston Dynamics, ETH Zurich, and others for real robot deployment). RSL-RL is specifically optimized for the Isaac Lab ecosystem.

### PyTorch + CUDA

**What it is**: The deep learning framework and NVIDIA's GPU programming toolkit.

**Why we chose it**: This isn't really a choice — it's the gravitational center of the ML ecosystem. Isaac Lab, RSL-RL, and virtually every modern RL library are built on PyTorch. CUDA is the only viable GPU compute framework for this kind of work (sorry, AMD).

### Paramiko (SSH from Python)

**What it is**: A Python library for SSH connections.

**Why we chose it**: We needed to run commands on the H100 server from a Windows laptop. PowerShell's SSH is fine for simple commands, but it mangles complex bash one-liners — dollar signs get interpreted as PowerShell variables, redirects get intercepted, and quotes turn into a nesting nightmare. Paramiko sends the command string *verbatim* to the server's bash shell, bypassing Windows shell interpretation entirely.

**The analogy**: PowerShell SSH is like playing telephone through two translators. Paramiko is a direct line.

### Screen (Terminal Multiplexer)

**What it is**: A Linux utility that lets you run processes in detachable sessions.

**Why we chose it**: SSH connections drop. Networks hiccup. Laptops go to sleep. If your training run dies because your WiFi blinked, that's not engineering — that's amateur hour. `screen` runs the process in a persistent session on the server. You can disconnect, reconnect, check on it, and it keeps running regardless of what your laptop is doing.

### Conda (Environment Management)

**What it is**: A package and environment manager for Python.

**Why we chose it**: Isaac Sim requires specific versions of Python (3.11), PyTorch (2.7.0+cu126), and dozens of other packages. Conda isolates these dependencies so they don't conflict with anything else on the system. On the H100 server, we use `env_isaaclab`. On the local laptop, `isaaclab311`. Each environment is a clean, reproducible bubble.

---

## 7. Results: The Numbers Tell a Story

### Chart 1: Throughput Scaling — Steps Per Second vs. Environment Count

```
Steps/s (thousands)
 70 ┤                                                              ■ H100 (65K)
    │                                                         ····/
 60 ┤                                                    ····/
    │                                               ····/
 50 ┤                                          ····/
    │                                    ·····/
 40 ┤                               ■···/                          ← H100 hits 100% GPU
    │                          ■···/
 35 ┤                     ■···/
    │                ····/
 30 ┤           ····/
    │      ····/
 25 ┤ ····/                                                        ← H100
 20 ┤/
    │
 15 ┤·····■                                                        ← Local peaks here
    │····/ ·····■
 10 ┤··/         ·····■                                            ← Local (RTX 2000 Ada)
  8 ■
    │
  0 ┼──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────
    1K        2K        4K        8K       16K       32K       64K
                              Number of Environments
```

**What this tells us**:

The H100 curve has **three distinct phases**:

1. **Linear ramp (1K → 8K)**: Throughput scales almost linearly. Double the environments, nearly double the steps/s. The GPU is hungry for more work — like a V12 engine cruising in second gear.

2. **Plateau (8K → 16K)**: Throughput barely changes (36,447 → 37,865). The GPU compute units are saturated. Adding more environments just means each iteration takes longer without processing more steps per second. This is the GPU "hitting the rev limiter."

3. **Second wind (16K → 65K)**: Throughput *explodes* again to 69,090 steps/s. This is surprising. What's happening is that at very high env counts, the PPO algorithm gets more efficient — each learning update uses a much larger batch of experience, and the GPU's massive parallelism on the neural network side kicks in. The H100's 96 GB of HBM3 memory can keep all 65K environments in fast memory simultaneously.

**The local RTX 2000 Ada** barely scales past 2K envs. At 4K, it's already at 93% utilization and only managing 13K steps/s. It simply doesn't have the compute units or memory bandwidth to keep up.

### Chart 2: VRAM Usage — Memory Is Not the Bottleneck

```
VRAM Used (GB)
 96 ┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ H100 VRAM LIMIT (96 GB)
    │
 80 ┤
    │
 60 ┤                                      ← 63% headroom remaining at 65K envs!
    │
 40 ┤                                                              ■ 35.3 GB
    │
 20 ┤                                ■ 21.0 GB
    │                   ■ 13.4 GB
 10 ┤          ■ 10.2 GB
  8 ┤   ■ 7.9 ── ── ── ── ── ── ── ── ── ── ── RTX 2000 Ada LIMIT (8 GB)
  6 ┤■ 6.0                                                ■ 5.5 GB (local 4K)
    ■                                               ■ 4.2
    ■                                         ■ 3.6
  0 ┼──────┬──────┬──────┬──────┬──────┬──────┬──────
    1K    2K    4K    8K   16K   32K   64K
                    Number of Environments

    ■ H100        ■ RTX 2000 Ada       ── Capacity Limits
```

**What this tells us**:

The H100 used only **37% of its VRAM** at maximum load. We're nowhere near OOM. The theoretical OOM point (extrapolating the ~0.46 MiB/env slope) would be around **196,000 environments** — but the physics engine would collapse long before that.

The RTX 2000 Ada, meanwhile, used 67% of its VRAM at just 4K envs. Projecting forward, 8K envs would need ~7.8 GB, dangerously close to its 8 GB limit. **The local GPU's ceiling is around 6K–8K environments**, after which it'd OOM.

### Chart 3: GPU Temperature and Power — The Thermal Story

```
Temperature (°C) / Power (W)
 400 ┤── ── ── ── ── ── ── ── ── ── ── ── ── ── H100 TDP (400W)
     │
 300 ┤                                                     ■ 299W ← H100 Power
 250 ┤                                ■ 250W
 200 ┤                   ■ 204W
 170 ┤          ■ 171W
 146 ┤   ■ 146W
     │
 110 ┤■ 110W
     │
  83 ┤── ── ── ── ── ── ── ── ── ── ── Throttle point (~83°C)
  68 ┤   · · · · · · · · · · · · ·· ·○ 68°C ← Local Temp (4K)
  65 ┤                                          □ □ □ □ □ □ ○ 65°C ← H100 Temp
  58 ┤○ 58°C
     │□ 40°C
  40 ┤
     │         ▲ 38.4W ← Local GPU *exceeded* its 35W TDP at 4K envs!
  35 ┤── ── ── ── ─▲─ ── ── ── ── ── ── RTX 2000 Ada TDP (35W)
  30 ┤▲ 29.5W  ▲ 29.9W
     │
   0 ┼──────┬──────┬──────┬──────┬──────┬──────┬──────
     1K    2K    4K    8K   16K   32K   64K

     ■/□ H100 (Power/Temp)    ▲/○ Local (Power/Temp)
```

**What this tells us**:

The H100 is **remarkably cool** — peak 65°C is a full 18°C below the thermal throttle point. At 299W, it's using only 75% of its power budget. This GPU could run sustainably at this load 24/7 without breaking a sweat. In server terms, we're "cruising" — not "red-lining."

The local GPU, however, tells a different story. At 4K envs, it hit **38.4W — exceeding its rated 35W TDP**. This is the GPU's boost behavior — it temporarily exceeds its power limit for short bursts. During sustained training, this would cause thermal throttling and reduced clock speeds. The 68°C temperature is comfortable-ish for a laptop, but there's very little headroom left.

### Chart 4: The Speedup Gap Widens with Scale

```
H100 Speedup vs. Local RTX 2000 Ada
  
  8x ┤                                                              
     │                                                         ■ 7.7x
  7x ┤                                                    ···/
     │                                               ···/
  6x ┤                                          ···/      (65K vs local 4K peak)
     │                                     ···/
  5x ┤                                ···/                ← At this point,
     │                           ■··/                       the local GPU
  4x ┤                      ···/                            physically can't
     │                 ···/                                 run these env counts
  3x ┤            ···/
     │       ···/
  2x ┤  ■··/  1.85x ← Largest comparable test (both ran 4K envs)
     ■·/ 1.56x
  1x ■ 1.06x  ← At 1K envs, they're nearly identical!
     │
  0x ┼──────┬──────┬──────┬──────┬──────┬──────┬──────
     1K    2K    4K    8K   16K   32K   64K
```

**The punchline**: At 1,024 environments, the $30,000 H100 is only **6% faster** than the $2,000 laptop GPU. You'd never justify that cost for small-scale experiments.

But at 4,096 environments (the laptop's practical limit), it's **1.85x faster**. And the H100 can keep going — all the way to 65K envs and 69K steps/s. Comparing peak-to-peak, the H100 delivers **5.2x the throughput** of the local GPU's best effort.

The H100 earns its keep not through raw speed at small scale, but through the ability to **scale to places consumer hardware simply cannot go**.

---

## 8. The Bug Journal: What Broke and How We Fixed It

Every engineering project has a graveyard of bugs. Here are ours, and the lessons each one taught us.

### Bug #1: "The Conda Ghost" — Script Fails Silently on the Server

**What happened**: We deployed `stress_test.sh` to the H100, launched it in a `screen` session, and it immediately ran all 7 tests with exit code 1. Every single training run "completed" in under a second. The results were empty.

**Root cause**: The script started with `conda activate env_isaaclab`, which works fine in an interactive shell. But `screen` (and non-interactive bash generally) doesn't source `.bashrc`, so `conda` wasn't on the PATH. The `conda activate` line silently failed, meaning Python wasn't available, meaning Isaac Lab couldn't start.

**The fix**: Add this line *before* `conda activate`:
```bash
source /home/t2user/miniconda3/etc/profile.d/conda.sh
```
This manually initializes conda's shell integration.

**The lesson**: **Never assume your interactive shell environment matches your script environment.** When you type `conda activate` in your terminal, it works because your `.bashrc` already ran `conda init`. Scripts launched by `screen`, `cron`, `systemd`, or any other non-interactive context don't load `.bashrc`. Always explicitly source your dependencies.

Think of it like packing for a trip. In your house, you know where everything is. But when you ship a box to someone else, you need to pack *everything* it needs — it can't reach into your closet.

---

### Bug #2: "The mpstat Phantom" — CPU Metrics Were All Zeros

**What happened**: After the stress test completed, the CPU utilization column in `combined_results.csv` showed `0` for every single test. The `cpu_metrics.txt` files were all 64 bytes — just the error message.

**Root cause**: `mpstat` (part of the `sysstat` package) wasn't installed on the H100 server. The script launched `mpstat 1 > cpu_metrics.txt 2>&1 &` as a background process, which wrote "command not found" and exited. The subsequent `awk` parsing found no data and returned 0.

**The fix**: For immediate results, we used the GPU metrics (which were complete) and the RAM metrics. For future runs, install `sysstat`:
```bash
sudo apt install sysstat
```

**The lesson**: **Always verify your dependencies exist on the target machine *before* running your experiment.** A 30-second `which mpstat` check would have caught this. More importantly, the script should have validated its tools at startup:
```bash
command -v mpstat >/dev/null 2>&1 || { echo "ERROR: mpstat not found. Install sysstat."; exit 1; }
```

This is called "fail fast" and it's one of the most important principles in engineering. Better to crash loudly at the start than silently produce garbage.

---

### Bug #3: "The Warp Wall" — Training Gets Stuck for 30 Minutes

**What happened**: Our very first training attempt on the H100 appeared to hang completely. The GPU showed 0% utilization but the CPU was at 100%. No output for over 30 minutes.

**Root cause**: NVIDIA Warp (the kernel compilation framework used by Isaac Sim) JIT-compiles GPU kernels the first time they're used. Rough terrain with 1,024 environments requires generating procedural terrain with 3.5 million triangle faces. The kernel for this had never been compiled before — so Warp was spending 30+ minutes compiling it on the CPU before anything could run on the GPU.

**The fix**: Run a small test (4 environments) first to populate the Warp kernel cache. Subsequent runs with any number of environments reuse the cached kernels and start in ~30 seconds.

**The lesson**: **JIT compilation is invisible until it hurts you.** If you're using any framework that compiles code at runtime (Warp, Numba, JAX, PyTorch's `torch.compile`), your first run will be dramatically slower than all subsequent runs. Factor this into your benchmarking — and always do a "warm-up" run first. Record it in your experimental methodology so reviewers know you did it right.

---

### Bug #4: "Death by Signal" — Accidentally Killing Training

**What happened**: During the 30-minute Warp compilation, we sent `SIGUSR1` to the process to check its status (a common debugging technique for many Linux programs). The process immediately died.

**Root cause**: Isaac Sim doesn't handle `SIGUSR1`. The default action for unhandled signals in Linux is to terminate the process. We killed our own training.

**The lesson**: **Don't send signals to processes you don't fully understand.** Instead, use non-intrusive monitoring — check log files, use `nvidia-smi`, check `/proc/<pid>/status`. And for long-running processes, always use `screen` or `tmux` so you can safely detach and reattach without affecting the process.

---

### Bug #5: "The RAM Metric Lie" — Parsing the Wrong Column

**What happened**: The `combined_results.csv` showed ~1,031,732 MiB of RAM used for every test. That's 1 TiB — the *total* RAM, not the *used* RAM.

**Root cause**: The `ram_metrics.csv` format was: `timestamp total used free available`. Our awk command was grabbing field `$2` (total) instead of field `$3` (used). A classic off-by-one error in column indexing.

**The fix**: We wrote a PowerShell script locally to re-parse the raw `ram_metrics.csv` files with the correct column index and regenerated the combined results.

**The lesson**: **Always validate your parsing against the actual data format.** `head -1 ram_metrics.csv` would have shown us the column layout immediately. In data pipelines, a single wrong column index can make your entire analysis worthless — and it'll *look* plausible enough that you might not notice.

---

### Bug #6: "The PowerShell Saboteur" — Shell Character Conflicts

**What happened**: We tried to run bash one-liners on the H100 server using PowerShell's SSH:
```powershell
ssh t2user@server "for d in 1024 2048; do echo $d; done"
```
PowerShell intercepted `$d` as a PowerShell variable (empty), `2>/dev/null` directed output to `C:\dev\null` (which doesn't exist), and the whole command was mangled.

**The fix**: We used `h100_run.py` (a Paramiko-based SSH script) to send commands verbatim to the server's bash shell, completely bypassing PowerShell's string interpretation.

**The lesson**: **Windows PowerShell and bash have fundamentally incompatible string interpretation rules.** Dollar signs, backticks, redirects, and quotes all mean different things. When working across OS boundaries, use a dedicated SSH library (Paramiko, Fabric) or write your complex logic in a script file and `scp` it to the server, rather than trying to inline it through SSH.

---

### Bug #7: "The One-SSH Rule" — Server Crashes on Multiple Connections

**What happened**: Early in the project, opening a second SSH session to the H100 caused the server to crash and require a physical reboot.

**Root cause**: The server had an unstable network configuration — `systemd-networkd-wait-online` was causing boot hangs, and the network stack couldn't handle concurrent connections. This is a configuration issue specific to our server, not a general Linux problem.

**The fix**: Disabled and masked `systemd-networkd-wait-online`, commented out a broken swap entry in `/etc/fstab`, and established a **strict rule: one SSH session at a time**.

**The lesson**: **Shared hardware requires shared discipline.** When multiple people use the same server, establish operational rules and document them prominently. Our "ONE SSH SESSION AT A TIME" rule prevented several potential crashes. It feels restrictive, but it's better than a 30-minute "walk to the server room and physically reboot it" interruption.

---

### Bug #8: "The Missing Mesa" — Headless Rendering Crash

**What happened**: Isaac Sim crashed immediately when trying to run in headless mode on the H100.

**Root cause**: Missing `libglu1-mesa`, an OpenGL utility library required even for headless rendering (because the EGL initialization path still references it).

**The fix**: 
```bash
sudo apt install libglu1-mesa
```

**The lesson**: **"Headless" doesn't always mean "no graphics dependencies."** GPU-accelerated simulators often need OpenGL/EGL libraries even when rendering to GPU memory without a display. Always check the installation docs for headless-specific requirements, and don't assume `--headless` means you can skip graphics packages.

---

## 9. Lessons for Future Engineers

### How Good Engineers Think

Building this project wasn't just about running scripts. Here are the meta-skills that made the difference:

#### 1. Plan Before You Code

We wrote `STRESS_TEST_PLAN.md` *before* writing a single line of `stress_test.sh`. The plan included:
- What we're testing and why
- What metrics we'd collect
- What we expected to find
- How we'd organize the output

This plan survived first contact with reality almost intact. The only changes were the bugs listed above — and because we had a plan, we knew immediately when something deviated from expectations.

**Anti-pattern**: Starting by writing code and "seeing what happens." You'll spend more time debugging than if you'd spent 20 minutes thinking first.

#### 2. Automate from the Start

We didn't manually run 7 separate training commands and manually copy metrics. We wrote one script that ran all 7 tests, collected all metrics, and generated a summary CSV. The total human intervention after pressing "enter" was zero.

**Why this matters**: When you automate, you get reproducibility for free. Anyone can re-run our stress test on any machine and get comparable results. When you do things manually, you introduce human error, you can't repeat it exactly, and you can't hand it off.

#### 3. Monitor Everything

We collected GPU utilization, VRAM usage, temperature, power draw, clock speed, RAM usage, and training metrics — all sampled every second. Most of this data was "boring" and confirmed what we expected. But the RAM metric bug (#5) was caught *because* we had the data, and the PhysX overflow at 65K envs was visible *because* we logged stderr.

**The rule**: Collect more data than you think you need. Storage is cheap. Re-running experiments is expensive.

#### 4. Version Control Non-Negotiable

Every result, every config, every script was committed to git and pushed to GitHub. The 100-iteration test was commit `577b9a6`. The 1-hour training results were commit `c35acfb`. The stress test results were the commit after that.

**Why**: Three months from now, when someone asks "what exact configuration produced that 69,090 steps/s result?", we can point to a specific commit with every file in the exact state it was in when that number was generated. That's not bureaucracy — that's science.

#### 5. Fail Fast, Fail Loud

Our biggest time-wasters were silent failures — conda not activating (no error), mpstat not installed (error swallowed by redirect), wrong column parsed (output looked plausible). The best debugging investment is making failures *obvious*:

```bash
# Bad: Fails silently
conda activate my_env

# Good: Fails loudly  
source /path/to/conda.sh || { echo "FATAL: conda init failed"; exit 1; }
conda activate my_env || { echo "FATAL: conda activate failed"; exit 1; }
```

### Best Practices We Followed (and You Should Too)

| Practice | Why It Matters |
|---|---|
| **Screen sessions for long tasks** | Network drops won't kill your 6-hour training run |
| **PYTHONUNBUFFERED=1** | Logs appear in real-time, not after the process exits |
| **10-second cooldown between tests** | Lets GPU clocks and thermals settle to baseline |
| **Fresh process per test** | No warm-GPU-state carryover contaminating results |
| **Warp kernel pre-cache** | Avoids 30-min JIT compilation mid-benchmark |
| **SCP results immediately** | Don't leave data on a server that might crash |
| **Document configs with YAML** | Exact reproducibility of training hyperparameters |
| **Git push after every milestone** | Work is never more than one commit from safety |

---

## 10. Recommendations for Maximizing Training Runs

Based on our results, here's a decision framework:

### If You're on the H100:

```
                    ┌─────────────────────────┐
                    │ What's your priority?    │
                    └───────────┬─────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌──────────────┐ ┌──────────┐ ┌───────────────┐
        │ Fast debug/   │ │ Balanced │ │ Max throughput │
        │ iteration     │ │ training │ │ (overnight)    │
        └──────┬───────┘ └────┬─────┘ └───────┬───────┘
               ▼              ▼               ▼
        ┌──────────────┐ ┌──────────┐ ┌───────────────┐
        │ 1,024 envs   │ │ 4,096 -  │ │ 32,768 envs   │
        │ 9K steps/s   │ │ 8,192    │ │ 41K steps/s   │
        │ Low VRAM     │ │ envs     │ │ 100% GPU      │
        │ 36% GPU      │ │ 24-36K   │ │ Clean physics │
        │              │ │ steps/s  │ │               │
        │ Good for:    │ │ Best     │ │ Good for:     │
        │ - Testing    │ │ steps/s  │ │ - Production  │
        │   code       │ │ per watt │ │   policies    │
        │ - Quick      │ │          │ │ - Overnight   │
        │   reward     │ │ Good for:│ │   runs        │
        │   checks     │ │ - Long   │ │               │
        │              │ │   runs   │ │ AVOID 65K:    │
        │              │ │ - Multi- │ │ PhysX drops   │
        │              │ │   hour   │ │ contacts →    │
        │              │ │   training│ │ bad physics   │
        └──────────────┘ └──────────┘ └───────────────┘
```

**Our top recommendation: 8,192 environments for most training runs.**

Why? It hits the sweet spot of throughput (36K steps/s), efficiency (4.07x speedup over 1K), and resource use (10 GB VRAM, 65% GPU, 49°C). You have massive headroom for longer/harder tasks, the physics are clean, and the hardware is comfortable. It's like driving at 70% throttle on a highway — fast, efficient, sustainable.

**Do NOT use 65,536 environments for production training.** Yes, the throughput number looks incredible (69K steps/s), but the PhysX `collisionStackSize` overflow means contacts are being dropped. Your robot is "learning" on broken physics. The resulting policy might look good in simulation but fail on hardware because it learned to exploit ghost collisions.

### If You're on the RTX 2000 Ada (or similar laptop GPU):

| Scenario | Envs | Why |
|---|---|---|
| **Code testing** | 64–256 | Instant startup, verify nothing crashes |
| **Quick experiments** | 1,024 | 8.5K steps/s, reasonable iteration time |
| **Maximum effort** | 2,048–4,096 | 10-13K steps/s, but GPU is at its limit |

Don't try to compete with the H100 for training speed. Use the laptop for development and debugging, then deploy to the server for production runs.

### Time Projections for Full Training Runs

Using our 1-hour test as calibration (1,400 iterations, 1,024 envs, reward 10.76):

| Goal | Envs | Steps/s | Time for 34M steps | Time for 100M steps |
|---|---|---|---|---|
| H100 balanced | 8,192 | 36,447 | ~15.5 min | ~46 min |
| H100 max safe | 32,768 | 40,944 | ~13.8 min | ~41 min |
| Local max | 4,096 | 13,239 | ~42.8 min | ~2.1 hours |

At 8K envs on the H100, a run that takes 1 hour locally finishes in **~15 minutes**. For overnight training (say, 500M steps), that's **3.8 hours on H100** vs. **10.5 hours locally**.

---

## 11. Where We'll Hit Walls Next

Our stress test answered many questions, but it also revealed upcoming challenges:

### 1. PhysX Collision Stack Limits

**The problem**: At 65K envs, PhysX ran out of collision stack memory and started dropping contacts. This is configurable (you can increase `collisionStackSize` in the scene descriptor), but it requires modifying Isaac Lab internals — not just command-line flags.

**Recommendation**: If we need more than 32K envs with clean physics, we'll need to either:
- Increase the PhysX collision stack size (requires Isaac Lab code modification)
- Use a simpler terrain (flat ground needs far fewer collision vertices)
- Use a task with fewer contact-rich interactions

### 2. Multi-GPU Scaling

**The problem**: We only tested on a single H100. NVIDIA's NVLink technology allows multi-GPU training, where environments are split across GPUs. Isaac Lab supports this, but it introduces new bottlenecks — GPU-to-GPU communication, memory synchronization, and load balancing.

**Recommendation**: If the team gets access to a multi-GPU system, run the same stress test on 2× and 4× GPU configurations to find the multi-GPU scaling curve.

### 3. CPU Bottleneck at Scale

**The problem**: We couldn't collect CPU metrics because `sysstat` wasn't installed. But the Warp compilation bug tells us the CPU matters more than we think — terrain generation, policy serialization, and logging all happen on the CPU.

**Recommendation**: Install `sysstat`, re-run the stress test, and look for CPU saturation at high env counts. If the CPU becomes a bottleneck, consider using `taskset` to pin the training process to specific cores, avoiding contention with OS tasks.

### 4. Disk I/O for Checkpointing

**The problem**: Our 10-iteration tests didn't save checkpoints. But full training runs save model snapshots every N iterations. On the H100, the disk is a standard NVMe SSD. At 32K envs, each checkpoint could be large, and writing it to disk mid-training injects latency.

**Recommendation**: Monitor disk I/O during long training runs. If checkpoint writes cause visible latency spikes, consider:
- Saving checkpoints less frequently
- Using a RAM disk (`/dev/shm`) for temporary checkpoints
- Async checkpoint saving via a background thread

### 5. Network Stability for Remote Training

**The problem**: Our H100 server has a history of crashing on multiple SSH sessions. The network configuration is fragile. A crash during an overnight training run means lost work.

**Recommendation**:
- Always use `screen` or `tmux` (we already do this)
- Set up automatic checkpoint saving (Isaac Lab does this by default, every 50 iterations)
- Consider a `systemd` service to auto-restart crashed training runs
- Long-term: fix the network stack properly (replace `systemd-networkd` with `NetworkManager`, or debug the underlying NIC driver issue)

### 6. Systems That Could Make This Better

| System | What It Does | Why We'd Want It |
|---|---|---|
| **Weights & Biases (W&B)** | Cloud experiment tracking | Real-time loss curves, hyperparameter comparison, team dashboards — no more grepping log files |
| **TensorBoard** | Local training visualization | Isaac Lab already generates TensorBoard events; we just need to forward the port via SSH |
| **Prometheus + Grafana** | Infrastructure monitoring | Real-time GPU/CPU/RAM dashboards instead of post-hoc CSV parsing |
| **Docker containers** | Environment isolation | Ship the entire env as a container — no more "works on my machine" |
| **Kubernetes + NVIDIA GPU Operator** | Multi-job scheduling | Queue multiple experiments, auto-allocate GPU resources |
| **NFS or shared storage** | Shared model artifacts | Both machines access the same model files without SCP |

---

## 12. Final Thoughts

This project started as a simple question — *"how far can we push this GPU?"* — and turned into a comprehensive study of GPU-accelerated robotics simulation, infrastructure engineering, and the fine art of making things break gracefully.

The data tells a clear story: **the H100 is a monster, but it's a monster that needs to be driven well.** Running 1,024 environments on it is like using a firehose to water a houseplant. Running 32,768 is where it shines — 41K steps/s of clean physics, a comfortable 63°C, and 22% VRAM utilization. That's the hardware working as intended.

The RTX 2000 Ada surprised us too — mostly by *not* OOM-ing at 4,096 envs. It's a capable machine for development work, and the fact that it nearly matches the H100 at 1K envs means you can prototype locally with high confidence that your code will scale up.

But the real lessons aren't in the numbers. They're in the bugs. The conda ghost, the Warp wall, the RAM metric lie — these are the kinds of problems that eat hours if you're not prepared for them. The engineers who work fastest aren't the ones who type fastest. They're the ones who:

1. **Write a plan** before writing code
2. **Validate assumptions** before trusting results
3. **Automate everything** so mistakes happen once, not every time
4. **Log everything** so problems are diagnosable after the fact
5. **Fail loudly** so problems are *visible* in the first place

The stress test is complete. The data is in. Now we know exactly how to configure our training runs for maximum efficiency — and exactly where the walls are so we can plan around them.

**Time to teach some robots to walk.**

---

*Generated from stress test data collected on February 12, 2026. All results are reproducible using the scripts and configurations in this repository.*
