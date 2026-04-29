# Loco_Shared — Capstone Locomotion Shared Infrastructure

Common Python package used by all five Loco_Policy_N projects on this branch:

- `Loco_Policy_1_ARL_Baseline/`
- `Loco_Policy_2_ARL_Hybrid/`
- `Loco_Policy_3_Student_Teacher_Training/`
- `Loco_Policy_4_Expert_Master_Distilled/`
- `Loco_Policy_5_Final_Capstone_Policy/`

## Package layout

```
quadruped_locomotion/
├── utils/
│   ├── training_utils.py        — clamp_noise_std, register_std_safety_clamp, configure_tf32
│   ├── lr_schedule.py           — cosine_annealing_lr, set_learning_rate
│   └── dr_schedule.py           — domain randomization curriculum
├── tasks/
│   ├── locomotion/
│   │   └── mdp/
│   │       └── rewards.py       — clamped_action_smoothness_penalty, clamped_joint_acceleration_penalty,
│   │                              clamped_joint_torques_penalty, clamped_joint_velocity_penalty
│   │                              + base reward terms used across all 5 policies
│   └── navigation/              — nav-task wrappers (used by Loco_5)
├── robots/                      — robot articulation defs (Spot, ANYmal-C, Vision60)
└── ai_trainer/                  — AI Coach integration (used by Loco_2 ARL Hybrid)
```

## Usage

Each Loco_Policy_N's training script appends `Loco_Shared/` to sys.path
and imports as `from quadruped_locomotion.utils import ...`. See the
per-policy `scripts/train.py` for the exact path setup.

## Why "Shared"

These utilities + reward wrappers + base classes are inherited and reused
across the entire 5-policy ship matrix. Keeping them in one place avoids
import-path drift and ensures all five policies see identical defense
layers (Bug #25 watchdog, clamped L2 penalties, NaN sanitizers).
