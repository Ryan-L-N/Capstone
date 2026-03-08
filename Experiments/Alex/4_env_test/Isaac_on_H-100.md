# Running Isaac Sim & Isaac Lab on the H-100 Server

**Date:** February 12, 2026  
**Server:** `172.24.254.24` (hostname: `ai2ct2`)  
**Login:** `ssh t2user@172.24.254.24` — Password: `!QAZ@WSX3edc4rfv`

---

## Quick Reference

| Component       | Version       | Location                              |
|-----------------|---------------|---------------------------------------|
| Isaac Sim       | 5.1.0         | pip package in conda env              |
| Isaac Lab       | 0.54.3        | `/home/t2user/IsaacLab`               |
| Conda env       | env_isaaclab  | Python 3.11.14                        |
| PyTorch         | 2.7.0+cu126   | CUDA enabled                          |
| GPU             | H100 NVL      | 95830 MiB VRAM                        |
| NVIDIA Driver   | 580.126.16    | CUDA 13.0 driver / 13.1 toolkit      |

---

## Connecting to the Server

```bash
# From CMU network or VPN
ssh t2user@172.24.254.24
```

> **CRITICAL: Only ONE SSH session at a time.**  
> Multiple parallel SSH sessions have caused the server to become completely unresponsive.  
> Close any existing session before opening a new one.

---

## Activating the Environment

Every time you SSH in, activate the Isaac Lab conda environment:

```bash
conda activate env_isaaclab
```

This gives you:
- Python 3.11.14 (required by Isaac Sim 5.1.0)
- PyTorch 2.7.0 with CUDA support
- All Isaac Sim/Lab packages
- RL frameworks: rl_games, rsl_rl, sb3, skrl, robomimic

To verify everything is working:
```bash
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')"
python -c "import isaacsim; print('Isaac Sim OK')"
python -c "import isaaclab; print(f'Isaac Lab {isaaclab.__version__}')"
```

---

## Running Training (Headless)

All training on this server should run **headless** (no display). The H100 is a headless server with no monitor attached.

### Quick Test — Train an Ant

```bash
conda activate env_isaaclab
cd ~/IsaacLab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Ant-v0 \
    --headless
```

### Train a Quadruped Robot (Anymal-C)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Velocity-Rough-Anymal-C-v0 \
    --headless
```

### Train with a Specific RL Framework

**RSL-RL (default for locomotion):**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Velocity-Rough-Anymal-C-v0 \
    --headless \
    --num_envs 4096
```

**Stable Baselines 3:**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py \
    --task=Isaac-Ant-v0 \
    --headless \
    --num_envs 4096
```

**SKRL:**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task=Isaac-Ant-v0 \
    --headless \
    --num_envs 4096
```

**RL Games:**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task=Isaac-Ant-v0 \
    --headless \
    --num_envs 4096
```

---

## Key Training Parameters

| Parameter       | Description                                      | Example              |
|-----------------|--------------------------------------------------|----------------------|
| `--task`        | The task/environment to train                    | `Isaac-Ant-v0`       |
| `--headless`    | Run without display (required on this server)    | (flag, no value)     |
| `--num_envs`    | Number of parallel environments                  | `4096`               |
| `--max_iterations` | Max training iterations                       | `1000`               |
| `--seed`        | Random seed for reproducibility                  | `42`                 |
| `--video`       | Record video of evaluation                       | (flag, no value)     |
| `--video_length`| Length of video in steps                          | `200`                |

---

## Available Tasks

List all available tasks:
```bash
./isaaclab.sh -p scripts/tools/list_envs.py
```

Key locomotion tasks:
- `Isaac-Velocity-Rough-Anymal-C-v0` — Anymal-C on rough terrain
- `Isaac-Velocity-Flat-Anymal-C-v0` — Anymal-C on flat terrain
- `Isaac-Velocity-Rough-Anymal-D-v0` — Anymal-D on rough terrain
- `Isaac-Velocity-Rough-Unitree-Go2-v0` — Unitree Go2
- `Isaac-Velocity-Rough-Spot-v0` — Boston Dynamics Spot
- `Isaac-Ant-v0` — Simple ant (good for quick tests)
- `Isaac-Humanoid-v0` — Humanoid locomotion

---

## Running Evaluation / Playing a Policy

After training, evaluate a saved policy:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Isaac-Velocity-Rough-Anymal-C-v0 \
    --headless \
    --num_envs 32 \
    --load_run <run_directory_name>
```

The `--load_run` argument should be the name of the run directory inside `logs/rsl_rl/`.

To find your training runs:
```bash
ls -lt ~/IsaacLab/logs/rsl_rl/
```

---

## Long-Running Training with Screen

For training that takes hours/days, use `screen` so it survives SSH disconnections:

```bash
# Create a named screen session
screen -S training

# Inside the screen session:
conda activate env_isaaclab
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Velocity-Rough-Anymal-C-v0 \
    --headless \
    --num_envs 4096

# Detach from screen: press Ctrl+A, then D
```

**Reconnecting later:**
```bash
ssh t2user@172.24.254.24
screen -r training
```

**List all screen sessions:**
```bash
screen -ls
```

---

## Monitoring GPU During Training

While training is running (in another screen window or after detaching):

```bash
# One-shot GPU check
nvidia-smi

# Live monitoring (updates every second)
watch -n 1 nvidia-smi
```

You should see GPU utilization increase and memory usage go up when training is active.

---

## Training Logs & Checkpoints

Training outputs are saved in:
```
~/IsaacLab/logs/<framework>/<task>/<timestamp>/
```

For example:
```
~/IsaacLab/logs/rsl_rl/anymal_c_rough/2026-02-12_15-30-00/
├── model_100.pt        # Checkpoint at iteration 100
├── model_200.pt        # Checkpoint at iteration 200
├── model_last.pt       # Latest checkpoint
├── config.yaml         # Training configuration
└── tb/                 # TensorBoard logs
```

### Viewing TensorBoard Logs

From the server:
```bash
conda activate env_isaaclab
tensorboard --logdir ~/IsaacLab/logs/ --port 6006 --bind_all
```

Then from your local machine (on CMU network/VPN), open in browser:
```
http://172.24.254.24:6006
```

---

## Running Custom Scripts

To run your own Python scripts that use Isaac Sim/Lab:

```bash
conda activate env_isaaclab
cd ~/IsaacLab

# Use isaaclab.sh -p to run with the right Python
./isaaclab.sh -p /path/to/your_script.py --headless

# Or directly with python (conda env must be active)
python /path/to/your_script.py --headless
```

---

## Using Docker (Alternative)

The NVIDIA Container Toolkit is also installed if you prefer Docker:

```bash
# Test GPU access in Docker
sudo docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# Run Isaac Sim container (example)
sudo docker run --rm --gpus all \
    -e OMNI_KIT_ACCEPT_EULA=YES \
    nvcr.io/nvidia/isaac-sim:4.5.0 \
    ./isaac-sim.sh --headless
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'isaacsim'"
Make sure the conda environment is activated:
```bash
conda activate env_isaaclab
```

### CUDA out of memory
Reduce `--num_envs`:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Velocity-Rough-Anymal-C-v0 \
    --headless \
    --num_envs 1024  # reduce from 4096
```

### Server unresponsive
- Do NOT open multiple SSH sessions simultaneously
- If the server is unresponsive to SSH but pings, wait 5 minutes
- If still unresponsive, a physical hard reboot may be needed (hold power button)
- After reboot, all software persists — just `conda activate env_isaaclab` and resume

### Training killed unexpectedly
Check if the process ran out of memory:
```bash
dmesg | tail -20
```
Reduce `--num_envs` or check system RAM with `htop`.

### First run is very slow
The first time you run Isaac Sim, it downloads and caches extensions from the registry. This can take **10+ minutes**. Subsequent runs will be faster.

---

## Environment Summary

```
Server:     172.24.254.24 (ai2ct2)
OS:         Ubuntu 22.04.5 LTS
GPU:        NVIDIA H100 NVL (95830 MiB)
Driver:     580.126.16
CUDA:       13.1 (nvcc) / 13.0 (driver)
Docker:     29.2.1 (with --gpus all support)
Conda env:  env_isaaclab (Python 3.11.14)
Isaac Sim:  5.1.0
Isaac Lab:  0.54.3
PyTorch:    2.7.0+cu126
```

---

**Last Updated:** February 12, 2026
