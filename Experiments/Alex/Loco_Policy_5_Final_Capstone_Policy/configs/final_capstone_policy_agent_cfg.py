"""PPO runner config for parkour-nav training.

Arch matches SIM_TO_REAL expert (proven on V18), with two additions:
  1. observation history length = 10 (cribbed from Cheng 2024)
  2. asymmetric critic via `obs_groups` — Isaac Lab / RSL-RL 3.x native
     support. Actor sees `policy` ObsGroup; critic sees `policy + critic`.
"""

import os
import sys

_SIM2REAL_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "SIM_TO_REAL")
)
if _SIM2REAL_ROOT not in sys.path:
    sys.path.insert(0, _SIM2REAL_ROOT)

from configs.agent_cfg import SpotS2RExpertPPORunnerCfg
from isaaclab.utils import configclass


@configclass
class FinalCapstonePolicyPPORunnerCfg(SpotS2RExpertPPORunnerCfg):
    """PPO config for the unified teacher policy.

    Key deltas vs SIM_TO_REAL expert:
      - experiment_name -> spot_final_capstone_policy
      - num_steps_per_env 24 -> 32 (longer horizons due to nav waypoints)
      - obs_groups wires `critic` ObsGroup into the RSL-RL critic network
    """

    experiment_name: str = "spot_final_capstone_policy"
    run_name: str = "parkour_nav_v1"

    num_steps_per_env: int = 32
    max_iterations: int = 15000
    save_interval: int = 100

    # Asymmetric critic — actor sees noisy proprio + raycast; critic also
    # sees clean obs + friction + added_mass + foot contacts (privileged_obs).
    # RSL-RL auto-routes these into actor_critic.py when both keys exist.
    obs_groups: dict = {
        "policy": ["policy"],
        "critic": ["policy", "critic"],
    }

    # Observation history length — the S2R wrapper's obs_delay buffer is
    # repurposed to stack last N observations rather than delay a single one.
    obs_history_length: int = 10


# TODO: distillation runner cfg (Phase 2).
# Once the teacher (privileged critic + clean obs actor) is trained, we need a
# student that sees ONLY the noisy policy observations + a GRU belief encoder.
# Pattern: use RSL-RL's DistillationRunner OR write a BC-on-rollouts loop.
