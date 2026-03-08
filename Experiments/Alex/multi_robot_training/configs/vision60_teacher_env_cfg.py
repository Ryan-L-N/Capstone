"""Vision60 Phase 2a: Teacher Environment with Privileged Observations.

Extends Vision60PPOEnvCfg with privileged observation dimensions:
  - Terrain friction coefficient [from body_physics_material]
  - Per-foot contact force magnitudes [from contact_forces]

Template: hybrid_ST_RL/configs/teacher_env_cfg.py (adapted for Vision60)
Created for AI2C Tech Capstone — MS for Autonomy, February 2026
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from configs.vision60_ppo_env_cfg import Vision60PPOEnvCfg, Vision60PPOObservationsCfg


@configclass
class Vision60TeacherObservationsCfg(Vision60PPOObservationsCfg):
    """Teacher observations with privileged terrain information."""

    @configclass
    class PolicyCfg(Vision60PPOObservationsCfg.PolicyCfg):
        """Standard observations + privileged terrain info."""

        # Privileged: terrain friction
        terrain_friction = ObsTerm(
            func=mdp.body_physics_material,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="body")},
        )

        # Privileged: per-foot contact forces (clean, no noise)
        # Vision60 uses "lower.*" for foot bodies
        foot_contact_forces = ObsTerm(
            func=mdp.contact_forces,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="lower.*")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class Vision60TeacherEnvCfg(Vision60PPOEnvCfg):
    """Vision60 Phase 2a Teacher Environment with privileged observations.

    DR is fixed at final expanded values.
    """

    observations: Vision60TeacherObservationsCfg = Vision60TeacherObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Fix DR at final expanded values
        self.events.physics_material.params["static_friction_range"] = (0.1, 1.5)
        self.events.physics_material.params["dynamic_friction_range"] = (0.08, 1.2)
        self.events.push_robot.params["velocity_range"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
        self.events.push_robot.interval_range_s = (6.0, 13.0)
        self.events.base_external_force_torque.params["force_range"] = (-6.0, 6.0)
        self.events.base_external_force_torque.params["torque_range"] = (-2.5, 2.5)
