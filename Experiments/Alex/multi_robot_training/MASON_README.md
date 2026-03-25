# Mason — AI Coach VLM Camera Test Instructions

Hey Mason, we need you to verify the VLM frame capture on your RTX 6000 Ada (48GB).
My RTX 2000 Ada (8GB) segfaults when `--enable_cameras` loads the RTX rendering pipeline,
so this needs a beefier GPU.

---

## What You're Testing

The AI coach can receive **rendered camera frames** from the simulation for visual gait
analysis (VLM mode). We just upgraded the frame capture pipeline:

1. **Multi-env sampling** — samples from envs 0, 10, 50 and picks the best (brightest) frame
2. **Reset filtering** — skips envs mid-reset (episode < 10 steps) to avoid garbage frames
3. **Frame averaging** — buffers frames from the last 5 iterations before each coach check,
   averages them into a composite image for a more stable visual signal

We need to confirm the camera sensor actually produces non-black frames on your GPU.

---

## Setup

### 1. Conda Environment

You need Isaac Lab installed in a conda env. If you already have one (e.g., `isaaclab311`),
use that. Otherwise, follow the Isaac Lab install docs.

```bash
# Activate your Isaac Lab conda env
conda activate isaaclab311   # or whatever yours is called
```

### 2. Install the Package

```bash
cd Experiments/Alex/multi_robot_training
pip install -e source/quadruped_locomotion/
```

Verify it works:
```bash
python -c "import quadruped_locomotion; print('OK')"
```

### 3. Install Anthropic SDK

```bash
pip install anthropic
```

---

## Test 1: Camera Verification (No Coach, Quick Check)

This runs 30 iterations with vision enabled to confirm frames aren't black.
No API key needed — the coach is disabled.

```bash
conda activate isaaclab311

python scripts/rsl_rl/train_ai.py \
  --start_phase mason_hybrid --end_phase mason_hybrid \
  --num_envs 128 --save_interval 50 \
  --max_noise_std 1.0 \
  --enable_cameras --enable_vision \
  --no_coach \
  --headless
```

**What to look for:**
- It should start training without crashing (no segfault, no Vulkan errors)
- After ~30 iterations, kill it with `Ctrl+C`
- If you see `[AI-TRAIN] Coach camera added: 320x240 RGB, update every 10s sim-time`
  in the output, the camera was injected successfully

### If it crashes:
- If you get a segfault in `rtx.scenedb.plugin.dll` — the RTX rendering pipeline
  can't initialize. Try without `--headless` (with a display attached).
- If you get an OOM error — reduce `--num_envs` to 64.

---

## Test 2: Full AI Coach with Vision (Needs API Key)

This runs the AI coach with VLM enabled so the coach receives averaged frames.

```bash
conda activate isaaclab311

python scripts/rsl_rl/train_ai.py \
  --start_phase mason_hybrid --end_phase mason_hybrid \
  --num_envs 128 --save_interval 50 \
  --max_noise_std 1.0 \
  --coach_interval 10 --coach_mode immediate \
  --enable_cameras --enable_vision \
  --anthropic_api_key YOUR_KEY_HERE \
  --headless
```

**What to look for in the output:**
```
[AI-COACH] VLM frame: averaged 6 frames from envs [0, 10, 50]
[AI-COACH] iter=10 action=no_change confidence=0.XX latency=XXXXms
```

- `averaged 6 frames` = frame buffering is working (5 pre-buffered + 1 at coach time)
- If you see `averaged 1 frames` that's fine too — means only the final frame captured
  (camera `update_period=10s` sim-time may not have fired yet in early iters)
- The coach should respond with `no_change` in early iterations — that's correct behavior
- Let it run for ~30 iterations (3 coach checks at interval 10) then kill it

### If frames are all black:
- You'll see the coach still making decisions but without `VLM frame: averaged` messages
- This means the camera rendered but produced black pixels (lighting/render issue)
- Not a blocker — the coach works fine without vision, it's just a bonus feature

---

## Test 3: No-Vision Baseline (If Vision Crashes)

If `--enable_cameras` causes issues, verify the core training still works:

```bash
conda activate isaaclab311

python scripts/rsl_rl/train_ai.py \
  --start_phase mason_hybrid --end_phase mason_hybrid \
  --num_envs 128 --save_interval 50 \
  --max_noise_std 1.0 \
  --coach_interval 10 --coach_mode immediate \
  --anthropic_api_key YOUR_KEY_HERE \
  --headless
```

This should work on any GPU. Expected output:
```
[AI-COACH] iter=10 action=no_change confidence=0.XX latency=XXXXms
[AI-COACH] reason: Training just started...
```

---

## What to Report Back

1. **Did Test 1 crash?** (segfault, OOM, or clean?)
2. **Did you see `VLM frame: averaged N frames`?** (the key question)
3. **Any error messages** in the output (copy-paste anything with `[AI-TRAIN]` or `[AI-COACH]`)
4. **GPU info**: `nvidia-smi` output (just the GPU name and memory)

---

## Notes

- The `mason_hybrid` phase uses Mason's 11 reward terms + 3 safety additions (14 total)
  with a [512, 256, 128] network and adaptive KL learning rate schedule
- `--coach_mode immediate` makes the coach active from iter 0 (for quick testing).
  Production uses `deferred` mode with `--activation_threshold 300`
- The `--headless` flag is important — without it, Isaac Sim tries to open a display
- Kill training with `Ctrl+C` when you've seen enough. The process uses `os._exit(0)`
  on completion to avoid CUDA deadlocks
- Logs go to `logs/rsl_rl/spot_hybrid_ppo/<timestamp>/`
