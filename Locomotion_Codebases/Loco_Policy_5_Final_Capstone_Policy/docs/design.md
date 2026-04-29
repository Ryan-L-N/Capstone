# Loco_Policy_5_Final_Capstone_Policy — Unified Spot Policy Design Doc

**Goal:** One RL policy that passes **both** the 4-environment terrain test
(friction / grass / boulder / stairs ≤23cm risers) **and** Cole's dense
obstacle course (25 waypoints, 7 shape types). Sim2real hardening from iter 0.

**Team / Scope:** Alex + Capstone team, April–May 2026. One H100 shared with
Team7; budget ~3 weeks from kickoff.

---

## 1. Why a clean rebuild

V14–V19 stair experts plateau at **21m zone-3** (13cm risers). Root causes
diagnosed in `memory/stair_v19_training.md`:
- Altitude/directional_progress rewards at weight 3.0 + full resume
  fine-tune destroyed flat-ground gait.
- `action_scale=0.2` caps stride height below what 13cm risers need.
- No privileged critic → high value-function variance on stairs.
- No observation history → policy cannot compensate for 20-40ms
  real-hardware delays.
- Terrain curriculum trained specialists; specialists don't transfer to
  Cole's clutter (FM V3 flat dominates nav; rough experts fail it).

Fine-tuning cannot fix all of these simultaneously. Clean rebuild with
baked-in sim2real is faster than 10 more V2X attempts.

## 2. Architecture (Miki 2022 teacher-student, Cheng 2024 curriculum)

| Layer | Teacher (Phase 1) | Student (Phase 2, ships) |
|---|---|---|
| Policy obs | Mason-235 + 10-step history | Mason-235 + 10-step history |
| Critic obs | **privileged:** +friction, mass, contact forces, true terrain grid | (student drops critic) |
| Encoder | MLP [512, 256, 128] | **GRU belief encoder** [256] + MLP head |
| Action scale | **0.3** (was 0.2) | 0.3 |
| Training | PPO, 4096 envs, 8000 iters | BC + DAGGER on teacher rollouts, 6000 iters |

Why two stages: privileged critic converges 20–30% faster (Miki 2022, CoRL
2022 Agarwal). GRU belief encoder is the single biggest sim2real win for
noisy height_scan (dropout, per-ray noise). Student is what ships.

## 3. Training environment

**Terrain curriculum** (10 levels × 20 cols = 200 patches; game-curriculum):
- Stairs up (pyramid + HF) — risers ramp 3cm→23cm
- Boulders + random boxes — edges ramp 3cm→50cm
- Slopes (smooth + rough)
- Flat + stepping stones
- **Flat-clutter** with Cole-style scatter (7 shapes, 0–30 obstacles)

**Obstacle scatter on every terrain type** (the Cole twist): the policy
learns navigation around obstacles on stairs, boulders, slopes — not just
flat. This is what fuses 4-env-test competence with Cole competence.

**Commands:** `(vx, vy, ωz)` 3D from iter 0, resampled every 2–5s to mimic
A*/tangent planner step-changes. Command curriculum widens ranges with
terrain level.

**Domain randomization (parkour-calibrated):**
| Parameter | Range |
|---|---|
| Ground friction | 0.6–2.0 |
| Trunk mass Δ | 0–3 kg |
| COM offset | ±0.2 m |
| Motor Kp/Kd | ±20% |
| Action latency | 0–40 ms uniform |
| Pushes | 6–10s interval, 0.6 m/s |
| Proprio noise | dof_pos σ=0.01, dof_vel σ=1.5, grav σ=0.05 |
| Height-scan | σ=3cm per-ray + 8% dropout |

**Rewards:** pure velocity tracking + gait + survival. Exact terms copied
from `SIM_TO_REAL/S2RRewardsCfg` minus altitude/directional_progress.
Locked by design decision in `README.md` — do not re-litigate.

## 4. Three-week plan

**Week 1 — infra + teacher baseline**
- Wire FinalCapstonePolicyEnvCfg into gym registry (remaining TODOs in scaffold).
- Implement asymmetric critic obs group routing verification.
- Implement obstacle scatter module (reuse NAV_ALEX/online_obstacle_tracker).
- Launch teacher training, 2048 envs × 3000 iters as smoke test.
- Early eval against 4_env_test (expect poor stairs, OK friction/grass).

**Week 2 — teacher convergence + student distill**
- Full teacher run: 4096 envs × 8000 iters (~18h H100).
- Mid-training 4_env_test + Cole evals every 1000 iters.
- Launch student distillation: BC warmup 2000 iters + DAGGER 4000 iters.

**Week 3 — fine-tune, eval, ship**
- Fine-tune student on benchmark arenas (optional, low LR).
- Final 4_env_test and Cole rich/max eval suites.
- Document real-Spot deployment integration points (depth camera adapter,
  command ROS2 topic, kill switch).

## 5. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| 23cm riser still unreachable via curriculum alone | Medium | Week 3 fallback: add AMP motion prior with 1-2 MoCap stair clips |
| action_scale 0.3 causes jitter | Medium | action_smoothness weight bumped to -1.5 |
| Obstacle scatter on rough terrain blows up physics | Low | Start with box-only scatter; validate before adding meshes |
| RSL-RL wrapper doesn't route critic obs group | Medium | Verify in Week 1 smoke test; patch wrapper if needed |
| Cole dense-max still flaky | High | Accept: quarter-density Cole is deployable; max is stretch |

## 6. Success criteria (hard gates)

**Teacher (end of Week 2):**
- 4_env_test stairs ≥ 22m (beats V19)
- Friction ≥ 49m (matches V18)
- Body flip rate < 6% across all terrains

**Student (end of Week 3):**
- 4_env_test all arenas within 2m of teacher
- Cole quarter-density: 20/25 waypoints, 0 falls (3-seed mean)
- Cole max-density: 5/25 waypoints, ≤1 fall (stretch)
- Zero altitude/directional_progress reward terms

## 7. What we are explicitly NOT doing

- No per-terrain experts + distillation pipeline (V14–V19 proved this brittle)
- No altitude or directional_progress rewards (V19 post-mortem)
- No blind policy → perceptive policy pipeline (perception from iter 0)
- No manual gait priors unless Week 3 AMP fallback triggers
- No real-Spot deployment inside the 3-week window (sim-only sign-off)

## 8. Open questions for team review

1. Compute allocation vs Team7 YOLO window (Apr 21–23) — start teacher
   Apr 24 or overlap?
2. Is the ~3cm V18 ceiling actually `action_scale`-limited or
   reward-limited? Quick A/B: retrain V18 baseline with action_scale=0.3
   as a single-variable control (2h smoke).
3. Do we ship teacher OR student if Week 2 student underperforms? Default:
   ship whichever beats V19 on 4_env_test AND hits ≥15/25 on Cole quarter.

---

*Owner: Alex. Review: Capstone team lead. Status: DRAFT 2026-04-20.*
