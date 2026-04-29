# Final Capstone Policy Handoff — Unified Parkour/Nav Policy for Cole

**From:** Alex / Gabriel
**Date:** 2026-04-24
**Branch:** `development`

This folder contains the latest version of the unified Spot locomotion policy
(`parkour_phase3_7000.pt`) built as "Final Capstone Policy" plus instructions for
running it through the Cole navigation arena. The policy replaces the prior
mason-family / FM V3 locomotion layers in your nav stack.

---

## What's in this folder

| File | Purpose |
|---|---|
| `parkour_phase3_7000.pt` | **The policy.** Locomotion checkpoint trained on parkour DR + 4-arena terrain curriculum + Phase-3 widened DR + harder terrain. Iter 7000 of 8100. |
| `README.md` | This file. Human-readable handoff. |
| `CLAUDE_CONTEXT.md` | Context brief for Claude Code / any LLM you give this to. Copy-paste it as project instructions so your AI knows what the policy is and how to run it. |
| `FINAL_CAPSTONE_POLICY_EXPLAINED.md` | Plain-language writeup of how this policy was built — the 6-option NaN hunt, reward hacking, fresh restart, Phase-3 DR widening. Useful context but not required to run. |

---

## Where the code lives (not duplicated here)

This handoff only contains the **checkpoint + docs**. The code to run it is
already in this repo — you don't need a separate install.

| What | Path |
|---|---|
| Locomotion policy wrapper + height-scan raycast + gain setup | `Experiments/Alex/4_env_test/src/spot_rough_terrain_policy.py` |
| 4-env arena eval (friction/grass/boulder/stairs) | `Experiments/Alex/4_env_test/src/run_capstone_eval.py` |
| Cole arena builder + skill-nav-lite navigator + A* planner + depth sensor | `Experiments/Alex/NAV_ALEX/scripts/cole_arena_skillnav_lite.py` |
| Unified eval launcher (both 4-env and Cole) | `Experiments/Alex/Loco_Policy_5_Final_Capstone_Policy/scripts/eval.py` |
| Training config (env, agent, terrain DR) | `Experiments/Alex/Loco_Policy_5_Final_Capstone_Policy/pn_cfg/` |
| Training script (if you want to fine-tune further) | `Experiments/Alex/Loco_Policy_5_Final_Capstone_Policy/scripts/train.py` |

Your Cole-arena navigator modules (APF, online map, A* planner, depth
raycast) live under `Experiments/Alex/NAV_ALEX/source/nav_locomotion/` —
unchanged from the current main-line stack.

---

## How to run it

**Environment:** conda env `isaaclab311` (Python 3.11 + Isaac Lab / Isaac Sim 5.1.0).

### 4-env arena (friction / grass / boulder / stairs)
From the Capstone repo root:
```bash
python Experiments/Alex/Loco_Policy_5_Final_Capstone_Policy/scripts/eval.py \
    --checkpoint Experiments/Cole/Final_Capstone_Policy_handoff/parkour_phase3_7000.pt \
    --target 4_env \
    --envs friction,grass,boulder,stairs \
    --num_episodes 1
```
Add `--headless` for no-render. The launcher already bakes in
`--mason --action_scale 0.3` and spawn z=0.55 which this policy needs.

### Cole arena — onboard-realistic (RECOMMENDED for real-robot claims)
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
`--depth_sensor` implies `--online_map` (forward raycast + occupancy grid +
A* replans every 2s). This is the **deployment-realistic** pipeline —
no ground-truth obstacle cheat.

### Cole arena — quarter density (the proven ship-gate)
Drop the density params for the skill-nav-lite proven recipe:
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

---

## What this policy is good at (and not)

| Arena | Iter 7000 result | Notes |
|---|---|---|
| **Friction** (low friction floor) | FLIP 41.8m zone 5 / 108s | Reaches the *last* zone — furthest progress any ckpt has made. Not yet a clean COMPLETE but close. |
| **Grass** (drag zones) | **COMPLETE 49.5m / 212s** | Fastest grass run of the series. |
| **Boulder** (obstacle field) | FLIP 33.0m zone 4 / 158s | First ckpt to break into zone 4 (+10m over all prior). |
| **Stairs** (up to 23cm riser) | FLIP 29.3m zone 3 / 114s | +5m past the V14–V17–V18 wall that had blocked stairs policies for months. |
| **Cole TRUE max** (full clutter 1/1/1) | **7/25 WPs, 0 falls, 514m, 6m45s** | Previous best: stock_flat 5/25 on easier arena (no small-static). Evaluated on iter 6800 with same pipeline; iter 7000 expected similar or better. |

**Strengths:**
- Unified policy — one file handles all four arena types
- Clean gait (no hopping; straight walk at commanded speed)
- Navigates Cole max density without falling over a 6+ minute run
- Tracks velocity commands tightly (error_vel_xy ≈ 0.4)
- Doesn't freeze in uncertain terrain (earlier iter 4000 did; this one pushes through)

**Limitations:**
- 4-env eval COMPLETE count is 1/4 (only grass clean-completes; the others FLIP near the end). Progress distance is dramatically better, but goal reach isn't yet consistent.
- Low-friction eval arena coefficient may be below the 0.4 training floor; expect some residual FLIPs on very slippery surfaces until a wider-DR fine-tune.
- Cole score system drains over time; max-density runs will hit `SCORE_DEPLETED` before all 25 WPs even if nav is correct. Normal.

---

## Key design facts (short version)

- **Observation:** 235-dim (48 proprioceptive + 187 height scan via PhysX raycast)
- **Action:** 12 joint position targets (hip-x ×4, hip-y ×4, knee ×4), scaled by `action_scale=0.3`
- **Control rate:** 50 Hz (decimation=10 at 500 Hz physics)
- **PD gains:** Kp=60, Kd=1.5, solver_pos=4, solver_vel=0 (matches Isaac Lab SPOT_CFG)
- **DOF ordering:** type-grouped `[hx×4, hy×4, kn×4]` — NOT leg-grouped. If you reset joints to a default pose, use the pattern in `_FALLBACK_DEFAULT_POS` in `spot_rough_terrain_policy.py`.
- **Spawn height:** z=0.55 (lower causes leg clipping with the type-grouped default pose; higher causes drop-impact)

---

## If things go wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| Robot spawns deformed / falls on first frame | Wrong DOF default pose order (leg-grouped vs type-grouped) | Use `_FALLBACK_DEFAULT_POS` from `spot_rough_terrain_policy.py` verbatim |
| Robot stands frozen, doesn't step | Height-scan returning wrong values (e.g. reading robot body as ground) | Verify robot prim is `/World/Spot` and self-hit filter matches (fixed Apr 18 — see `raycast_solver_bugs_apr18.md` in Gabriel's notes) |
| Robot hops on grass | Undertrained gait; not a crash but expected from early ckpts | This is iter 7000; the hop from iter 2500 is gone. If you see it, verify you're loading the 7000 ckpt not an earlier one |
| Cole eval reactive-only (no depth sensor) flips quickly | APF alone can't handle dense clutter | Always include `--depth_sensor` for max-density tests |
| Isaac Sim hangs on exit | Known Isaac Sim 5.1 issue | Scripts use `os._exit(0)` to avoid it. Don't use `simulation_app.close()` directly |

---

## Questions / things we'd be curious to see

- Does a proprio-only student distillation from this teacher preserve the walking behavior at max Cole density? (Phase 3 ran teacher-only, student distill was cut for time)
- Does fine-tuning with an **AMP motion prior** (MoCap trot clip) kill the remaining variance in gait under slippery conditions?
- At quarter-density Cole (the proven skill-nav-lite recipe), how close to 25/25 does iter 7000 get?

Ping Alex or Gabriel with what you find.
