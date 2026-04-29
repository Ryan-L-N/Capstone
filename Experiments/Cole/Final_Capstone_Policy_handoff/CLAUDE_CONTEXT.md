# Claude Context — Final Capstone Policy Handoff

*Paste this into your Claude Code session or add it to a CLAUDE.md so the AI has
full context on what this policy is, where the code lives, and how to run it.
The goal: let Claude help you evaluate, fine-tune, or deploy without re-deriving
the project's history.*

---

## The mission (READ FIRST)

**Your job is to reproduce Alex/Gabriel's success on Cole max-density using
`parkour_phase3_7000.pt`.**

The target result (achieved on the sibling ckpt iter 6800 — iter 7000 is
strictly better on the 4-env battery and is expected to match or exceed this):

> **Cole TRUE max-density (all three obstacle categories at 1.0, onboard depth
> sensor + online mapping + A* replan):**
> **7/25 waypoints reached, 0 falls, 514.1m walked, 6m 45s of continuous
> navigation. Ended on `SCORE_DEPLETED` (score budget ran out) — NOT a crash.**

Prior project records this beats:
- stock_flat: 5/25 (on easier arena, no small-static hazards)
- mason_hybrid: 3/25
- V6 rough: 0/25

This run used 358 obstacles including 200 small-static trip hazards that no
prior max-density eval had included. Policy walked through all of them without
falling.

**Your first move:** run the exact command in the "Launch commands → Cole,
onboard-realistic" section below on `parkour_phase3_7000.pt` and confirm you
hit ≥5/25 WPs with 0 falls. If you match or beat 7/25, log the result.

**If you see fewer WPs or any fall on a trip hazard, troubleshoot in this
order:**
1. Verify `--depth_sensor` is in the command (it implies `--online_map`; A*
   replans on discovered obstacles). The reactive-only pipeline FELL at 1/25
   because APF in dense clutter cancels its own repulsion vectors.
2. Confirm `--rough_heightscan` is set (height scan obs mode, 187 dims).
   Without it the policy gets wrong observations.
3. Confirm `--loco_action_scale 0.3 --loco_decimation 1` — these MUST match
   the training config. Scale 0.2 produces a different actor distribution.
4. Check spawn + joint defaults are the type-grouped order (see below). Wrong
   order spawns the robot deformed and it falls immediately.
5. Check seed. Our result was seed 42. Different seeds give different
   obstacle placements and random-push timing.

---

## What this policy is

`parkour_phase3_7000.pt` is a PPO-trained locomotion policy for Boston Dynamics
Spot (12 joints, type-grouped DOF order `[hx×4, hy×4, kn×4]`). It was built in
April 2026 as "Final Capstone Policy" — a 5.5-day compressed build of what was
originally a 3-week unified-policy design. It replaces the prior per-arena
policies (V14–V19 for stairs; stock_flat / stock_rough for flat; mason family
for navigation).

**One brain for four arenas:** friction (low-mu floor), grass (drag zones),
boulder (obstacle field), stairs (up to 23cm risers). Trained on parkour-paper
domain randomization + custom 10-column terrain curriculum.

**Training history:**
- Fresh-start teacher (Apr 23): iter 0 → 6000, 4096 envs, reward 253 at end
- Phase-3 fine-tune (Apr 24): resumed from iter 6100 with widened DR
  (friction 0.6–2.0, mass 0–3kg, push ±0.6 m/s) and harder terrain (stairs
  to 30cm, boulder to 60cm, slope to 29°, +40% obstacle density)
- Iter 7000 (this ckpt): 900 Phase-3 iters in. Best 4-env progression in the
  series.

**Not a deployed policy yet.** 4-env eval shows 1/4 COMPLETE (grass). The
other three arenas FLIP near the end — progression dramatically better than
prior ckpts (friction zone 5, boulder zone 4, stairs +5m past V18 wall) but
not yet clean reaches.

---

## Observation and action space

- **obs_dim = 235**: 48 proprioceptive (joint pos, joint vel, root ang vel,
  gravity projection, last actions, commands) + 187 height scan (PhysX
  raycast over a 1.6m×1.0m patch under the robot, clipped to [-0.2, 0.3])
- **act_dim = 12**: joint position targets, scaled by `action_scale=0.3`,
  added to default standing pose
- **Control rate = 50 Hz** (decimation=10 at 500 Hz physics)
- **PD gains**: Kp=60, Kd=1.5, solver_pos=4, solver_vel=0

**Critical DOF order:** Isaac Sim's Spot USD uses type-grouped ordering
`[fl_hx, fr_hx, hl_hx, hr_hx, fl_hy, fr_hy, hl_hy, hr_hy, fl_kn, fr_kn, hl_kn, hr_kn]`.
**NOT** leg-grouped. Default standing pose is:
```python
_FALLBACK_DEFAULT_POS = np.array([
     0.1, -0.1,  0.1, -0.1,   # hx: fl, fr, hl, hr
     0.9,  0.9,  1.1,  1.1,   # hy: fl, fr, hl, hr
    -1.5, -1.5, -1.5, -1.5,   # kn: fl, fr, hl, hr
], dtype=np.float64)
```
Spawn z = **0.55** (lower = legs clip ground on reset; higher = drop impact).

---

## Where the code lives

All paths relative to Capstone repo root:

| Component | Path |
|---|---|
| Policy wrapper (loads ckpt, runs forward pass, handles height scan) | `Experiments/Alex/4_env_test/src/spot_rough_terrain_policy.py` |
| 4-env eval driver (spawn, stabilize, run N eps, collect metrics) | `Experiments/Alex/4_env_test/src/run_capstone_eval.py` |
| Unified eval launcher (calls 4-env or Cole suites with Final-Capstone-Policy-correct flags) | `Experiments/Alex/Loco_Policy_5_Final_Capstone_Policy/scripts/eval.py` |
| Cole arena + navigator (APF + online map + A* + depth sensor) | `Experiments/Alex/NAV_ALEX/scripts/cole_arena_skillnav_lite.py` |
| Navigator modules (skill_nav_lite, online_obstacle_tracker, grid_astar_planner, depth_raycast_detector) | `Experiments/Alex/NAV_ALEX/source/nav_locomotion/nav_locomotion/modules/` |
| Training configs (env, agent, terrain DR) | `Experiments/Alex/Loco_Policy_5_Final_Capstone_Policy/pn_cfg/` |
| Training entrypoint | `Experiments/Alex/Loco_Policy_5_Final_Capstone_Policy/scripts/train.py` |
| Privileged observations (critic-only: true friction, mass, foot forces) | `Experiments/Alex/Loco_Policy_5_Final_Capstone_Policy/modules/privileged_obs.py` |

**Reference copies of the exact nav-stack files used for the 7/25 result are
included under `Experiments/Cole/Final_Capstone_Policy_handoff/code/`** — see
`code/CODE_INDEX.md`. Those are for inspection / diff against future changes.
**Do NOT run from the handoff copies** — they're standalone files and the
scripts expect the `nav_locomotion` package layout at the canonical paths
above. Run from the canonical paths; the handoff `.pt` is accessible via
`--loco_checkpoint Experiments/Cole/Final_Capstone_Policy_handoff/parkour_phase3_7000.pt`.

---

## Launch commands

### Environment
```bash
conda activate isaaclab311   # has Isaac Sim 5.1.0, Isaac Lab, RSL-RL, PyTorch-CUDA
```

### 4-env eval (friction / grass / boulder / stairs)
```bash
python Experiments/Alex/Loco_Policy_5_Final_Capstone_Policy/scripts/eval.py \
    --checkpoint Experiments/Cole/Final_Capstone_Policy_handoff/parkour_phase3_7000.pt \
    --target 4_env \
    --envs friction,grass,boulder,stairs \
    --num_episodes 1
```
Launcher bakes in `--mason --action_scale 0.3` + type-grouped DOF + spawn z=0.55.
Add `--headless` to skip rendering. Results land in
`results/parkour_nav_eval/{env}_rough_episodes.jsonl`.

### Cole eval, onboard-realistic (recommended)
```bash
python Experiments/Alex/NAV_ALEX/scripts/cole_arena_skillnav_lite.py \
    --loco_checkpoint Experiments/Cole/Final_Capstone_Policy_handoff/parkour_phase3_7000.pt \
    --loco_action_scale 0.3 --loco_decimation 1 \
    --cole_arena --rough_heightscan --episodes 1 --seed 42 \
    --moveable_pct 1.0 --nonmoveable_pct 1.0 --small_static_pct 1.0 \
    --apf_radius 1.5 --apf_gain 0.9 --apf_tangent 0.8 \
    --max_lin_speed 2.4 --waypoint_reach 0.9 \
    --depth_sensor --rendered
```
Key flag: `--depth_sensor` uses a forward-facing PhysX raycast (8m, 90°/30°
FOV, 64×16 rays, 0.4m grid). Implies `--online_map` — robot discovers
obstacles, A* replans on growing map every 2s.

### Cole eval, quarter density (skill-nav-lite proven recipe)
```bash
python Experiments/Alex/NAV_ALEX/scripts/cole_arena_skillnav_lite.py \
    --loco_checkpoint Experiments/Cole/Final_Capstone_Policy_handoff/parkour_phase3_7000.pt \
    --loco_action_scale 0.3 --loco_decimation 1 \
    --cole_arena --rough_heightscan --episodes 1 --seed 42 \
    --moveable_pct 0.25 --nonmoveable_pct 0.25 --small_static_pct 0.0 \
    --apf_radius 1.5 --apf_gain 0.9 --apf_tangent 0.8 \
    --max_lin_speed 2.4 --waypoint_reach 0.9 \
    --depth_sensor --rendered
```

### Continue training (Phase-3 fine-tune pattern)
```bash
python Experiments/Alex/Loco_Policy_5_Final_Capstone_Policy/scripts/train.py \
    --phase teacher --headless \
    --num_envs 4096 --max_iterations 2000 --save_interval 100 \
    --seed 42 --max_noise_std 0.5 --lr_max 3e-4 --no_wandb \
    --resume_path Experiments/Cole/Final_Capstone_Policy_handoff/parkour_phase3_7000.pt
```
**IMPORTANT:** `--max_iterations` in RSL-RL is the *increment* to run, NOT the
target iter. `--max_iterations 2000` on a resume from iter 7000 ends at iter
9000 (RSL-RL's current_learning_iteration is loaded from the ckpt).

---

## The reward stack (high-level)

Positive (sum ~+3.0/step when policy is walking correctly):
- `base_linear_velocity` (weight 10) — forward progress
- `base_angular_velocity` (weight 5) — yaw tracking
- `gait` (weight 3) — alternating foot contact
- `foot_clearance` (weight 0.5) — feet lift over terrain
- `air_time` (weight 2) — foot-air time in [0.3, 0.8]s range

Negative:
- `action_smoothness` (weight -1.5) — penalize jerky joint targets
- `undesired_contacts` (weight -1.0) — body hitting terrain
- `motor_power`, `joint_pos`, `joint_vel`, `joint_acc`, `torque_limit` — cost/limit penalties
- `base_motion` (weight -0.5) — roll/pitch motion

**Do not** bump the positive gait/air_time weights. That was the "Option 6
failure" — policy found a "stand still and jiggle" local min farming those
bonuses without tracking velocity. The fresh-restart rebalance dropped them
back to low values specifically to prevent this.

---

## Domain randomization (Phase-3, current ckpt)

| Parameter | Range |
|---|---|
| Static friction | 0.6 – 2.0 |
| Dynamic friction | 0.4 – 1.8 |
| Added base mass | 0 – 3.0 kg |
| Push interval | 6 – 10 s |
| Push velocity | ±0.6 m/s in x, y |
| Motor Kp/Kd scale | 0.8 – 1.2 (per-env, reset-time) |
| Sensor dropout | 8% on height scan rays |
| Action delay | 2 steps (40 ms @ 50 Hz) |
| Obs delay | 1 step (20 ms @ 50 Hz) |

**The friction range floor of 0.6 is still above some test-arena floors** —
the 4-env friction arena is around 0.3–0.4. If you see FLIPs on slippery
ground, the answer is another DR-widening fine-tune, not more training at
current DR. Evidence: iter 6000 (DR floor 0.8) → iter 7000 (DR floor 0.6)
moved friction progress from zone 4 (32m) to zone 5 (42m). Another widening
to floor 0.3 should close the gap.

---

## Terrain curriculum (current)

10 difficulty levels × 20 columns. Each cell is an 8×8 m patch.

| Terrain | Proportion | Hardest variant (level 9) |
|---|---|---|
| Pyramid stairs (mesh + HF) | 20% | 30 cm risers |
| Boulders + random boxes | 20% | 60 cm obstacles, 40 obstacles per patch |
| Slopes (smooth + rough) | 20% | 29° angle / 15 cm rough noise |
| Flat clutter (Cole-style) | 20% | 45 obstacles, 45 cm height |
| Stepping stones + flat | 20% | 25 cm gaps |

Curriculum promotes via `terrain_levels_vel` — advances an env's difficulty
level when the policy covers enough distance within the commanded velocity
budget. Current iter 7000 plateaus at level 5.4 (mean across envs).

---

## Things that will bite you

1. **Isaac Sim import order:** Create `AppLauncher` (via `isaacsim.SimulationApp`
   or `isaaclab.app.AppLauncher`) BEFORE importing any `omni.*` or
   `isaaclab.*` modules. Parse argparse BEFORE the launcher. Otherwise you
   get cryptic GPU-Foundation crashes.
2. **Never call `simulation_app.close()`.** It can hang in GPU driver
   cleanup and leave unkillable D-state zombies. Use `os._exit(0)` once
   you've saved your data.
3. **Height scan fill = 0.0, not 1.0.** The policy expects zero-fill for
   areas outside the scan patch on flat terrain (training range was
   [-0.000002, 0.148083]). Fill = 1.0 causes the policy to think it's about
   to step off a cliff every frame.
4. **Quaternion format: [w, x, y, z]** scalar-first throughout. Isaac Sim
   returns this; the policy expects this.
5. **DOF ordering:** type-grouped. If you reset joint positions manually,
   use the exact `_FALLBACK_DEFAULT_POS` from
   `spot_rough_terrain_policy.py`. Leg-grouped ordering will plant knee
   values in hx slots and the robot spawns deformed.
6. **Friction combine mode = "multiply"** in all environments.
7. **Python namespace-package shadowing:** our training configs live under
   `pn_cfg/` specifically because naming it `configs/` got shadowed by
   `SIM_TO_REAL/configs/`. If you add new config dirs, use unique names.
8. **Cole score system decays** at ~-0.74/s, recharges +50/WP. Max-density
   runs can hit `SCORE_DEPLETED` before reaching all 25 WPs even with a
   perfect nav — not a failure, just a score-budget termination.

---

## Known open questions

- **Friction gap**: iter 7000 still FLIPs on the low-friction arena.
  Hypothesis: eval arena μ < 0.6 training floor. Fix: Phase-4 fine-tune
  with static_friction_range=(0.3, 2.0).
- **Cole quarter-density gate**: not yet evaluated on iter 7000 (only on
  iter 6800, 7/25 at TRUE max density). Expected 20+/25 at quarter based
  on iter 6800's trajectory.
- **Student distillation**: the original Final Capstone Policy plan included a
  proprio-only student distilled from the privileged teacher. We shipped
  teacher-only. Whether the student preserves max-density Cole performance
  is unknown.
- **AMP motion prior**: stretch goal that was deprioritized. Would likely
  kill the small-amplitude hops that still sometimes appear under high-DR
  conditions.

---

## Who to ask

- Policy training history / NaN debugging / reward design: Alex (Gabriel)
- Cole arena, navigator, skill-nav-lite recipe: Alex
- Isaac Sim / Isaac Lab internals, USD weirdness: Alex
- Code review, pipeline review: whoever's on the capstone team rotation

See `FINAL_CAPSTONE_POLICY_EXPLAINED.md` (same folder) for the full 1,600-word
plain-language story of how this policy got built, including the bugs.
