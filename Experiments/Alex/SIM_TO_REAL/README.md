# SIM_TO_REAL: 6-Terrain-Expert Distillation Pipeline

Train 6 terrain-specialist Spot policies from scratch, each dominating one terrain
type with full sim-to-real hardening, then distill into a single deployment-ready
generalist that crushes all 4 eval environments.

**Target:** 49.5 m (5/5 zones) on Friction, Grass, Boulder, AND Stairs.

---

## Architecture

```
                         ┌─────────────────────────────────┐
                         │     6 TERRAIN EXPERTS            │
                         │  (trained from scratch, 50 Hz)   │
                         │                                  │
                         │  [1] Friction    [4] Boulders    │
                         │  [2] Stairs Up   [5] Slopes      │
                         │  [3] Stairs Down [6] Mixed Rough  │
                         └────────────┬────────────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────────────┐
                         │   LEARNED ATTENTION ROUTER       │
                         │   235 → 64 → 6 (softmax)        │
                         │   Routes obs to best expert      │
                         └────────────┬────────────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────────────┐
                         │   DISTILLED STUDENT POLICY       │
                         │   [512, 256, 128] MLP            │
                         │   Trained at 20 Hz               │
                         │   All S2R wrappers active         │
                         └────────────┬────────────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────────────┐
                         │   REAL SPOT DEPLOYMENT            │
                         │   spot_sdk_wrapper.py             │
                         │   safety_layer.py                 │
                         │   20 Hz control loop              │
                         └─────────────────────────────────┘
```

---

## Quick Start

### 1. Train all 6 experts on H100

```bash
cd ~/SIM_TO_REAL
screen -S s2r_train
bash scripts/train_all_experts.sh
```

Or train one expert at a time:

```bash
python scripts/train_expert.py \
    --expert_type stairs_up \
    --headless --no_wandb \
    --num_envs 4096 \
    --max_iterations 10000 \
    --save_interval 100 \
    --max_noise_std 0.5
```

### 2. Evaluate experts

```bash
python scripts/eval_expert.py \
    --expert_type stairs_up \
    --checkpoint checkpoints/expert_stairs_up/best.pt \
    --headless --num_episodes 10
```

### 3. Run distillation

```bash
python scripts/train_distill_s2r.py \
    --expert_friction checkpoints/expert_friction/best.pt \
    --expert_stairs_up checkpoints/expert_stairs_up/best.pt \
    --expert_stairs_down checkpoints/expert_stairs_down/best.pt \
    --expert_boulders checkpoints/expert_boulders/best.pt \
    --expert_slopes checkpoints/expert_slopes/best.pt \
    --expert_mixed_rough checkpoints/expert_mixed_rough/best.pt \
    --headless --no_wandb \
    --num_envs 4096 \
    --max_iterations 8000
```

### 4. Evaluate distilled student

```bash
python scripts/eval_student.py \
    --checkpoint checkpoints/student/best.pt \
    --headless --num_episodes 100
```

### 5. Export for deployment

```bash
python deploy/export_onnx.py --checkpoint checkpoints/student/best.pt
```

---

## The 6 Experts

| # | Expert | Terrain Split | Reward Overrides | Goal |
|---|--------|--------------|-----------------|------|
| 1 | Friction | 80% friction plane (mu 0.05-1.5) + 20% flat | foot_slip -1.5 | 5/5 friction zones |
| 2 | Stairs Up | 40% pyramid_stairs_up + 40% hf_stairs_up + 20% flat | foot_clearance 2.0, orientation -2.0, joint_pos -0.3 | 5/5 stairs ascending |
| 3 | Stairs Down | 80% pyramid_stairs_down + 20% flat | foot_clearance 2.0, orientation -2.0, joint_pos -0.3 | 5/5 stairs descending |
| 4 | Boulders | 40% boxes + 40% discrete_obstacles + 20% flat | foot_clearance 2.5, orientation -2.0, joint_pos -0.3 | 5/5 boulder zones |
| 5 | Slopes | 35% slope_up + 35% slope_down + 10% wave + 20% flat | foot_slip -1.5, orientation -2.5 | All slopes 0-30 deg |
| 6 | Mixed Rough | 40% random_rough + 40% stepping_stones + 20% flat | foot_clearance 1.5, gait 12.0, joint_pos -0.5 | Precise foot placement |

All experts share:
- **Network:** [512, 256, 128] MLP with ELU (Mason baseline)
- **PPO:** Adaptive KL, 5 epochs, 4 mini-batches, init_noise_std=1.0
- **S2R hardening:** Action delay 40 ms, obs delay 20 ms, sensor noise + 5% dropout,
  torque limits, motor power penalty, push forces, wider DR

---

## Sim-to-Real Hardening

Every expert trains with these mitigations from step 0:

| Feature | Value | Risk Addressed |
|---------|-------|---------------|
| Action delay | 40 ms (2 steps @ 50 Hz) | R1: Actuator latency |
| Observation delay | 20 ms (1 step @ 50 Hz) | R1: Sensor latency |
| Height scan dropout | 5% rays zeroed | R10: Sensor dropout |
| IMU drift | Ornstein-Uhlenbeck process | R5: Correlated noise |
| Observation noise | ±0.2 m height, ±0.2 m/s vel | R5: Idealized sensors |
| Motor power penalty | -0.005 weight | R7: Energy efficiency |
| Torque limit penalty | -0.3 weight (hip 45 Nm, knee 100 Nm) | R6: Motor limits |
| External pushes | ±3.0 N every 7-12 s | R8: Disturbances |
| Mass randomization | ±5.0 kg | R8: Mass variation |
| Friction randomization | 0.15-1.3 static | R8: Surface variation |

See `RISK_MATRIX.md` for full 10-risk analysis.

---

## Distillation

The distilled student combines all 6 experts via learned attention routing:

- **Router:** Small MLP (235 → 64 → 6 softmax) learns which expert to trust per observation
- **Loss:** `alpha * (MSE + 0.1 * KL)` on router-blended expert actions + PPO reward
- **Alpha annealing:** 0.8 → 0.2 over 8000 iters (expert-heavy → PPO-heavy)
- **Control rate:** 20 Hz (decimation = 25) — matches real Spot SDK
- **Terrain:** Balanced all-terrain mix (11 types, equal proportion)

---

## Deployment

```
deploy/
├── spot_sdk_wrapper.py      # Policy I/O ↔ Spot SDK bridge (20 Hz)
├── safety_layer.py          # E-stop, torque/orientation/velocity watchdogs
├── height_scan_builder.py   # Depth camera → 187-dim elevation grid
├── calibration.py           # Joint zero, PD gains, friction, latency
├── telemetry.py             # JSONL logging + UDP stream to laptop
├── export_onnx.py           # PyTorch → ONNX conversion
└── DEPLOYMENT_CHECKLIST.md  # 5-stage testing protocol
```

**5-Stage Testing Protocol:**
1. Tethered flat ground (lab) — standing, walking, turning
2. Tethered rough ground (lab) — foam mats, wood boards, plastic sheet
3. Untethered flat ground (outdoor) — 5 min continuous run
4. Untethered rough terrain (outdoor) — grass, gravel, curbs
5. Full course (matching 4-env eval) — physical friction/grass/boulder/stairs

---

## Project Structure

```
SIM_TO_REAL/
├── README.md                    # This file
├── PLAN.md                      # Detailed implementation plan
├── RISK_MATRIX.md               # 10-risk sim-to-real analysis
├── configs/                     # Environment configurations
├── wrappers/                    # S2R wrappers (delay, noise)
├── rewards/                     # New reward terms (power, torque)
├── distillation/                # 6-expert router + loss
├── scripts/                     # Training and eval scripts
├── deploy/                      # Real hardware deployment
├── checkpoints/                 # Trained models
└── logs/                        # TensorBoard logs
```

---

## Training Rules

All scripts enforce these rules (learned from 35 bugs across 12+ trials):

- `--max_noise_std 0.5` always explicit (Bug #28d: defaults to 1.0)
- Cosine LR annealing: 1e-3 → 1e-5 with 50-iter warmup
- `--save_interval 100` (100 iters ~ 65M steps between saves)
- NaN sanitizer on every forward pass (Bug #24: `clamp_()` doesn't fix NaN)
- All penalty rewards clamped to bounded ranges (Bug #29)
- Value loss watchdog: halves LR when value_loss > 100 (Bug #25)
- `os._exit(0)` at end (CUDA deadlock prevention)
- `--headless` on H100, `--no_wandb` on H100

---

## Success Criteria

| Metric | Target | Current Best |
|--------|--------|-------------|
| Friction | 49.5 m (5/5) | 49.5 m (5/5) |
| Grass | 49.5 m (5/5) | 49.5 m (5/5) |
| Boulder | **49.5 m (5/5)** | 30.4 m (4/5) |
| Stairs | **49.5 m (5/5)** | 15.7 m (2/5) |
| Composite gauntlet | **600/600** | ~200/600 |
| Flip rate | 0% | 0% |
| Torque compliance | 95% within limits | Not measured |
| 20 Hz stability | All terrains | Not tested |

---

*AI2C Tech Capstone — MS for Autonomy, March 2026*
*Created for sim-to-real transfer of RL quadruped locomotion policies*
