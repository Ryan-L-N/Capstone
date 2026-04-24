# PARKOUR_NAV

Unified locomotion + navigation policy for Boston Dynamics Spot.

**Target:** one policy that passes BOTH
1. 4-environment terrain test (friction / grass / boulder / stairs up to 23cm risers)
2. Cole's dense obstacle course (25 waypoints, 7 shape types, ~1km)

**Pedigree:** clean-slate rebuild informed by V14-V19 graveyard. Design cribbed from
Miki 2022 (Science Robotics) teacher-student, Cheng 2024 (ICRA) Extreme Parkour
curriculum, legged_gym DR ranges. Sim2real baked in from iteration 0.

**Status:** SCAFFOLD — no training launched. See `docs/design.md` for the plan.

## Directory

```
PARKOUR_NAV/
├── scripts/
│   └── train_parkour_nav.py          # entry point (not yet wired to H100)
├── configs/
│   ├── parkour_nav_env_cfg.py        # env: obs groups, DR, commands, rewards
│   ├── parkour_nav_terrain_cfg.py    # unified curriculum (flat→parkour)
│   └── parkour_nav_agent_cfg.py      # PPO: arch, lr, observation history
├── rewards/                           # (symlink / reuse SIM_TO_REAL/rewards/)
├── envs/                              # (arena blending, obstacle scatter)
├── modules/                           # (reuse NAV_ALEX/modules/skill_nav_lite)
├── terrains/                          # (custom terrain generators if needed)
└── docs/
    └── design.md                      # 1-page design doc
```

## Key design commitments (locked — do not re-litigate)

1. **No altitude reward. No directional_progress reward.** V19 proved these
   destabilize flat-ground gait under fine-tune. Velocity tracking + terrain
   curriculum does the climbing work.
2. **action_scale 0.3** (up from 0.2). Extreme Parkour uses 0.5; 0.3 is a
   compromise for Spot's shorter legs.
3. **10-step observation history** (up from 0). Required for delay sim2real.
4. **Asymmetric actor-critic.** Critic sees true friction/mass/contact/terrain
   height; actor sees only proprio + noisy 187-ray scan.
5. **Game curriculum.** Promote terrain level if `dist > 0.8 * vx_cmd * T_ep`;
   demote if `< 0.4`. No altitude-as-reward.
6. **Obstacles on every terrain patch.** Cole-style clutter (0-30 objects,
   7 shape types) scattered on ALL terrain types — not just flat.
7. **3D command randomization** `(vx, vy, ωz)` from iter 0. Trained on synthetic
   planner-like commands (step changes every 2-5s).
