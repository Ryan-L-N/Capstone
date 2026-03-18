# Phase C Navigation Training Guide

Step-by-step instructions for training the hierarchical navigation policy (nav 10 Hz commanding a frozen loco policy at 50 Hz).

---

## 1. Prerequisites

Before starting, confirm the following are in place:

- **Isaac Lab** installed with GPU support (`isaac-sim` 2023.1.1+ or compatible).
- **RSL-RL** installed (`pip install rsl-rl`).
- **Phase B locomotion checkpoint** available. The recommended checkpoint is `ai_coached_v8_10600.pt` (Trial 11l, terrain 4.83, [1024, 512, 256] architecture). Copy it into `NAV_ALEX/checkpoints/`.
- **(Optional) Anthropic API key** for the AI coach. Export it as `ANTHROPIC_API_KEY` in your environment. Training works without it using the `--no_coach` flag.

---

## 2. Installation

```bash
cd NAV_ALEX
pip install -e source/nav_locomotion/
python -c "import nav_locomotion; print('OK')"
```

You should see `OK` printed. If you get `ModuleNotFoundError`, verify you ran `pip install -e` from the correct directory and that `source/nav_locomotion/setup.py` (or `pyproject.toml`) exists.

---

## 3. Step 1: Smoke Test (local, any GPU)

Run unit tests first to validate the code without needing Isaac Lab:

```bash
python scripts/rsl_rl/smoke_test.py --skip_env
```

Then run the full smoke test with a small number of environments:

```bash
python scripts/rsl_rl/smoke_test.py --headless \
    --loco_checkpoint checkpoints/ai_coached_v8_10600.pt
```

**Expected output:**
- 4 unit tests pass (config validation, reward shapes, loco wrapper load, observation space).
- 10 environment steps complete without NaN or crashes.
- Console prints `All smoke tests passed.`

If the full smoke test fails but `--skip_env` passes, the issue is likely the loco checkpoint path or an Isaac Lab installation problem.

---

## 4. Step 2: Phase C-0 Warm-up (no coach, flat terrain only)

This warm-up phase trains the nav policy on flat ground without the AI coach. The goal is to verify the training loop works end-to-end and the robot learns basic forward locomotion under nav control.

```bash
python scripts/rsl_rl/train_nav.py --headless --no_wandb --no_coach \
    --loco_checkpoint checkpoints/ai_coached_v8_10600.pt \
    --num_envs 512 --max_iterations 5000 --save_interval 100
```

**What to watch:**
- The robot should start moving forward within the first few hundred iterations.
- `mean_reward` should be increasing steadily.
- No NaN in any logged metric (reward, value loss, policy loss).

**Success criteria:** Mean forward distance greater than 5 meters by iteration 2000. If the robot is stationary or falling, check that the loco checkpoint loaded correctly (look for `Loaded frozen loco policy` in the console output).

Checkpoints are saved every 100 iterations to `logs/spot_nav_explore_ppo/`. You can safely stop training once the success criteria are met.

---

## 5. Step 3: Phase C-1 Full Training (with coach, full curriculum)

Once the warm-up confirms everything works, run full training with the AI coach enabled and terrain curriculum active.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python scripts/rsl_rl/train_nav.py --headless --no_wandb \
    --loco_checkpoint checkpoints/ai_coached_v8_10600.pt \
    --num_envs 512 --max_iterations 20000 --coach_interval 250
```

**Monitoring:**
- **TensorBoard:** Launch with `tensorboard --logdir logs/` and watch the `AI_Coach/`, `Reward_Weights/`, and `Reward_Contrib/` panels.
- **Coach decisions:** `tail -f logs/spot_nav_explore_ppo/coach_decisions.jsonl` to see real-time AI coach reasoning.

**What to watch for:**
- Terrain level should advance as the policy improves. If it stalls, the coach should adjust reward weights automatically.
- Body height should stay above 0.30 m. If it drops, the robot may be belly-crawling (the coach should auto-detect this and tighten the height penalty).
- Value loss should remain stable (below 100). Spikes above 100 indicate potential instability.

**Cost estimate:** Approximately $2--4 for a full run with Sonnet (~300 coach API calls at ~$0.01 each).

---

## 6. Step 4: Evaluation

After training completes, evaluate the final policy over 100 episodes:

```bash
python scripts/rsl_rl/play_nav.py \
    --nav_checkpoint logs/spot_nav_explore_ppo/model_final.pt \
    --loco_checkpoint checkpoints/ai_coached_v8_10600.pt \
    --num_envs 100 --num_episodes 100
```

This runs 100 independent episodes and reports aggregate metrics: mean distance traveled, survival rate, collision rate, and goal-reach rate. Compare these against the baseline loco-only policy to quantify the navigation layer's contribution.

---

## 7. H100 Deployment

Transfer the codebase to the H100 server and set up the environment:

```bash
scp -r NAV_ALEX/ t2user@172.24.254.24:~/NAV_ALEX/
```

Then SSH into the H100 and run:

```bash
ssh t2user@172.24.254.24
conda activate env_isaaclab
cd ~/NAV_ALEX
pip install -e source/nav_locomotion/
pip install anthropic
cp ~/multi_robot_training/checkpoints/ai_coached_v8_10600.pt checkpoints/
python scripts/rsl_rl/train_nav.py --headless --no_wandb \
    --loco_checkpoint checkpoints/ai_coached_v8_10600.pt \
    --num_envs 512 --coach_interval 250
```

**Important H100 notes:**
- The `--headless` flag is **required** after BMC reboots (no display server available).
- The `--no_wandb` flag is needed since wandb is not configured on the H100.
- Always check for D-state zombie processes before starting: `ps aux | grep python` and verify no stale Isaac Sim processes are holding GPU memory.

---

## 8. Monitoring Training

**TensorBoard** (run on local machine with SSH tunnel or on the server directly):

```bash
tensorboard --logdir logs/
```

Key panels and metrics to watch:

| Metric | Healthy Range | Warning Sign |
|---|---|---|
| `mean_reward` | Increasing over time | Flat or decreasing after 1000 iters |
| `terrain_level` | Advancing every ~500 iters | Stuck at same level for 2000+ iters |
| `body_height` | > 0.30 m | Dropping below 0.25 m (belly crawl) |
| `value_loss` | < 50 | Spikes above 100 (instability) |
| `noise_std` | 0.2--0.5 | Above 0.7 (too much exploration) |

**Coach decision log:**

```bash
tail -f logs/spot_nav_explore_ppo/coach_decisions.jsonl
```

Each line is a JSON object with the coach's reasoning, recommended weight changes, and guardrail outcomes.

---

## 9. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| NaN at early iterations | Learning rate too high | Lower LR with `--lr_max 5e-5` |
| Robot not moving | Loco checkpoint not loaded | Check console for `Loaded frozen loco policy`; verify checkpoint path |
| Belly crawling | Height penalty too weak | Coach should auto-detect; manually increase height penalty weight if running without coach |
| `No module nav_locomotion` | Package not installed | Run `pip install -e source/nav_locomotion/` from the `NAV_ALEX` directory |
| CUDA out of memory | Too many environments | Reduce `--num_envs` to 256 or 128 |
| D-state zombie processes | Isaac Sim CUDA deadlock | Full server reboot via BMC (`/bmc-reset` skill or Redfish API at 172.24.254.25) |
| Value loss explosion | Gradient instability | The LR watchdog should halve LR automatically; if not, stop and resume with lower `--lr_max` |
| Stale `__pycache__` | OneDrive sync artifacts | Delete with `find . -type d -name __pycache__ -exec rm -rf {} +` and retry |

---

## 10. Resume from Checkpoint

If training is interrupted or you want to continue from a saved checkpoint:

```bash
python scripts/rsl_rl/train_nav.py --headless --no_wandb \
    --loco_checkpoint checkpoints/ai_coached_v8_10600.pt \
    --resume logs/spot_nav_explore_ppo/model_5000.pt \
    --num_envs 512 --max_iterations 20000
```

**Before resuming, always verify the checkpoint is not corrupted:**

```python
import torch
ckpt = torch.load("logs/spot_nav_explore_ppo/model_5000.pt", map_location="cpu")
for k, v in ckpt.items():
    if torch.is_tensor(v) and torch.isnan(v).any():
        print(f"NaN found in {k}")
```

If any tensor contains NaN, do not resume from that checkpoint. Fall back to an earlier one.

**Note:** The `--max_iterations` flag sets the absolute iteration target, not a relative count. If resuming from iteration 5000 with `--max_iterations 20000`, training continues until iteration 20000 (15000 more iterations).

Add `--no_coach` if you want to resume without the AI coach, or add `--coach_interval 250` to re-enable it.
