"""
World factory - creates Isaac Sim World with standard physics settings.

Consolidates the World/physics setup duplicated across 15+ scripts.
Default settings match the validated configuration from grass experiments:
  - physics_dt = 1/500 (500Hz - critical for quadruped stability)
  - rendering_dt = 10/500 (50Hz)
  - stage_units_in_meters = 1.0
  - default ground plane with configurable friction

Usage:
    from core import create_world, WorldConfig
    world = create_world()  # defaults
    world = create_world(WorldConfig(ground_friction=0.9))
"""

from dataclasses import dataclass


@dataclass
class WorldConfig:
    """Configuration for World creation."""
    physics_dt: float = 1.0 / 500.0
    rendering_dt: float = 10.0 / 500.0
    stage_units_in_meters: float = 1.0
    ground_friction: float = 0.8
    ground_dynamic_friction: float = 0.8
    ground_restitution: float = 0.01
    ground_prim_path: str = "/World/Ground"


def create_world(config=None):
    """
    Create World with standard physics settings and ground plane.

    Args:
        config: WorldConfig instance (uses defaults if None)

    Returns:
        omni.isaac.core.World instance
    """
    from omni.isaac.core import World

    config = config or WorldConfig()

    print("Creating world...")
    world = World(
        physics_dt=config.physics_dt,
        rendering_dt=config.rendering_dt,
        stage_units_in_meters=config.stage_units_in_meters,
    )

    print(f"Creating ground plane (friction={config.ground_friction})...")
    world.scene.add_default_ground_plane(
        z_position=0,
        name="ground",
        prim_path=config.ground_prim_path,
        static_friction=config.ground_friction,
        dynamic_friction=config.ground_dynamic_friction,
        restitution=config.ground_restitution,
    )

    return world
