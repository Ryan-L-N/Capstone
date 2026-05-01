# Loco_Policy_5_Final_Capstone_Policy

Unified locomotion + navigation policy for Boston Dynamics Spot.

**Target:** one policy that passes BOTH
1. 4-environment terrain test (friction / grass / boulder / stairs up to 23cm risers)
2. Cole's dense obstacle course (25 waypoints, 7 shape types, ~1km)

**Pedigree:** clean-slate rebuild informed by V14-V19 graveyard. Design cribbed from
Miki 2022 (Science Robotics) teacher-student, Cheng 2024 (ICRA) Extreme Parkour
curriculum, legged_gym DR ranges. Sim2real baked in from iteration 0.

**Status:** **SHIPPED May 1 2026.** Canonical checkpoint
`parkour_phasefwplus_22100.pt`. The Phase-2 student distillation never
ran; this ships as the asymmetric-critic teacher.

- **Headline numbers (100-ep canonical eval, May 1):** friction 96%
  COMPLETE / grass 75% COMPLETE / boulder 0% (0 falls, z3 wedge) /
  stairs 0% (51 fell at z2-z3).
- **What works:** friction zone-5 ice traversal, grass zone-5,
  procedural pyramid stair climbing, wide DR robustness.
- **What doesn't:** boulder dense z3 push, stair fall reduction, FW
  USD staircase climb. Plus: any retrain from 22100 collapses to the
  level-0 trap (8 consecutive failed attempts documented).

**Read these docs in order:**
1. `docs/SHIP_DECISION.md` — what shipped, why, and the failed retrain
   archaeology
2. `docs/KNOWN_LIMITATIONS.md` — what 22100 can't do (concrete
   workarounds where they exist)
3. `docs/FUTURE_WORK.md` — research agenda for whoever picks this up
4. `docs/FINAL_CAPSTONE_POLICY_EXPLAINED.md` — project narrative

## Directory

```
Loco_Policy_5_Final_Capstone_Policy/
├── scripts/
│   └── train.py          # entry point (not yet wired to H100)
├── configs/
│   ├── final_capstone_policy_env_cfg.py        # env: obs groups, DR, commands, rewards
│   ├── final_capstone_policy_terrain_cfg.py    # unified curriculum (flat→parkour)
│   └── final_capstone_policy_agent_cfg.py      # PPO: arch, lr, observation history
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
