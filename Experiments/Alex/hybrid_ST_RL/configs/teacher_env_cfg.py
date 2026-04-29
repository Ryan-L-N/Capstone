"""Stage 2a: Teacher Environment Configuration with Privileged Observations.

Extends SpotFinetuneEnvCfg with an additional privileged observation group
containing information only available in simulation:
  - Terrain friction coefficient (1 dim)
  - Terrain type one-hot (12 dims)
  - Per-foot contact force magnitudes (4 dims)
  - Terrain slope at robot position (2 dims)
  Total: 235 standard + 19 privileged = 254 dims

The teacher uses this extra information to learn better terrain-adaptive
policies, which are then distilled to a standard-observation student.

Created for AI2C Tech Capstone — hybrid_ST_RL, February 2026
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from configs.finetune_env_cfg import SpotFinetuneEnvCfg, SpotFinetuneObservationsCfg


# =============================================================================
# Teacher Observations — standard 235 dims + 19 privileged dims = 254 total
# =============================================================================

@configclass
class SpotTeacherObservationsCfg(SpotFinetuneObservationsCfg):
    """Teacher observations with privileged terrain information.

    The policy group is expanded to include privileged information that
    is only available in simulation. This gives the teacher an advantage
    over the student, enabling it to learn better terrain-adaptive behavior.

    NOTE: The privileged dims are concatenated into the same policy group.
    This means the teacher's actor network has 254 input dims instead of 235.
    Weight surgery is needed to initialize from a 235-dim checkpoint.
    """

    @configclass
    class PolicyCfg(SpotFinetuneObservationsCfg.PolicyCfg):
        """Standard observations + privileged terrain info."""

        # Privileged: terrain friction (sampled from DR)
        # This gives the teacher knowledge of the current surface properties.
        terrain_friction = ObsTerm(
            func=mdp.body_physics_material,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="body")},
        )

        # Privileged: per-foot contact forces (clean, no noise)
        # These tell the teacher exactly how much force each foot is experiencing.
        foot_contact_forces = ObsTerm(
            func=mdp.contact_forces,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# =============================================================================
# Teacher Environment Config
# =============================================================================

@configclass
class SpotTeacherEnvCfg(SpotFinetuneEnvCfg):
    """Stage 2a Teacher Environment with privileged observations.

    Identical to SpotFinetuneEnvCfg except:
    - Observations include privileged terrain information
    - DR is fixed at the final Stage 1 values (no progressive expansion)
    """

    observations: SpotTeacherObservationsCfg = SpotTeacherObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Fix DR at final expanded values (no progressive schedule)
        self.events.physics_material.params["static_friction_range"] = (0.1, 1.5)
        self.events.physics_material.params["dynamic_friction_range"] = (0.08, 1.2)
        self.events.push_robot.params["velocity_range"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
        self.events.push_robot.interval_range_s = (6.0, 13.0)
        self.events.base_external_force_torque.params["force_range"] = (-6.0, 6.0)
        self.events.base_external_force_torque.params["torque_range"] = (-2.5, 2.5)
        self.events.add_base_mass.params["mass_distribution_params"] = (-7.0, 7.0)
        self.events.reset_robot_joints.params["velocity_range"] = (-3.0, 3.0)
