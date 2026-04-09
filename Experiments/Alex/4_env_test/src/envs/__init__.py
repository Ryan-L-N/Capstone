"""Environment builders for the 4-environment capstone evaluation.

Each module creates a 30m x 50m arena with 5 difficulty zones (10m each).

Usage:
    from src.envs import build_environment
    env = build_environment("friction", stage, cfg)
"""

from .friction_env import create_friction_environment, create_friction_v2_environment
from .grass_env import create_grass_environment
from .boulder_env import create_boulder_environment
from .stairs_env import create_stairs_environment
from .stairs_approach_env import create_stairs_approach_environment
from .simple_stairs_env import create_simple_stairs_environment

ENVIRONMENT_BUILDERS = {
    "friction": create_friction_environment,
    "friction_v2": create_friction_v2_environment,
    "grass": create_grass_environment,
    "boulder": create_boulder_environment,
    "stairs": create_stairs_environment,
    "stairs_approach": create_stairs_approach_environment,
    "simple_stairs": create_simple_stairs_environment,
}


def build_environment(env_name, stage, cfg):
    """Factory function to build an environment by name.

    Args:
        env_name: One of "friction", "grass", "boulder", "stairs"
        stage: USD stage from Omniverse
        cfg: Evaluation configuration
    """
    if env_name not in ENVIRONMENT_BUILDERS:
        raise ValueError(f"Unknown environment: {env_name}. Choose from: {list(ENVIRONMENT_BUILDERS.keys())}")
    return ENVIRONMENT_BUILDERS[env_name](stage, cfg)
