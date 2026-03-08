"""Spot Phase 2a: Teacher Environment with Privileged Observations.

Extends SpotPPOEnvCfg with 19 privileged observation dimensions:
  - Terrain friction coefficient (1 dim)  [from body_physics_material]
  - Per-foot contact force magnitudes (4 dims * 3 = 12 dims) [from contact_forces]
  Total observed: 235 standard + privileged dims (actual count depends on
  the mdp functions' output; friction returns variable dims, contact returns 12)

The teacher uses privileged information to learn better terrain-adaptive
policies, which are then distilled to a standard-observation student.

Template: hybrid_ST_RL/configs/teacher_env_cfg.py
Created for AI2C Tech Capstone — MS for Autonomy, February 2026
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from configs.spot_ppo_env_cfg import SpotPPOEnvCfg, SpotPPOObservationsCfg


@configclass
class SpotTeacherObservationsCfg(SpotPPOObservationsCfg):
    """Teacher observations with privileged terrain information."""

    @configclass
    class PolicyCfg(SpotPPOObservationsCfg.PolicyCfg):
        """Standard observations + privileged terrain info."""

        # Privileged: terrain friction
        terrain_friction = ObsTerm(
            func=mdp.body_physics_material,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="body")},
        )

        # Privileged: per-foot contact forces (clean, no noise)
        foot_contact_forces = ObsTerm(
            func=mdp.contact_forces,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class SpotTeacherEnvCfg(SpotPPOEnvCfg):
    """Spot Phase 2a Teacher Environment with privileged observations.

    DR is fixed at aggressive final values (no progressive expansion).
    """

    observations: SpotTeacherObservationsCfg = SpotTeacherObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Fix DR at final expanded values
        self.events.physics_material.params["static_friction_range"] = (0.05, 1.5)
        self.events.physics_material.params["dynamic_friction_range"] = (0.02, 1.2)
        self.events.push_robot.params["velocity_range"] = {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}
        self.events.push_robot.interval_range_s = (5.0, 12.0)
        self.events.base_external_force_torque.params["force_range"] = (-8.0, 8.0)
        self.events.base_external_force_torque.params["torque_range"] = (-3.0, 3.0)
