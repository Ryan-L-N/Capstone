"""Spot locomotion task — historical gym registrations were here.

After the Loco_Policy reorganization, the env_cfg + Locomotion-{Flat,
Robust,Play,Teacher,Distill}-Spot-v0 registrations live in
``Loco_Policy_1_ARL_Baseline/configs/__init__.py``, and the
mason_hybrid_env_cfg + Locomotion-ARLHybrid-Spot-v0 registrations live in
``Loco_Policy_2_ARL_Hybrid/configs/__init__.py``.

This package retains shared utilities (utils/, tasks/locomotion/mdp/,
robots/) consumed by all five Loco_Policy_N projects.
"""

# `from . import agents` was removed — each Loco_Policy_N now imports
# its own agent cfg from its own configs/agents/ directory.
