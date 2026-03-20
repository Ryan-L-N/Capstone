"""6-level curriculum terrain configuration for Phase C navigation.

Uses Isaac Lab's TerrainGeneratorCfg with curriculum=True.
6 difficulty rows x 10 terrain type columns = 60 terrain patches.
All robots start at level 1 (easiest) and are promoted/demoted based
on survival and forward velocity tracking.

Sub-terrains scale difficulty by row:
    Level 1 (easiest): smooth flat, tiny bumps, small boxes
    Level 6 (hardest): large boulders, steep stairs, high-amplitude waves

Terrain types and weights (10 types):
    flat (10%), random_rough (15%), boxes (15%), stairs_up (12%),
    stairs_down (8%), wave (10%), discrete_obstacles (10%),
    polyhedral_boulders (15%), friction_plane (5%), vegetation_plane (5%)
"""

from __future__ import annotations

from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.trimesh import (
    MeshPlaneTerrainCfg,
    MeshRandomGridTerrainCfg,
    MeshRepeatedBoxesTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
)
from isaaclab.terrains.height_field import (
    HfWaveTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
)


NAV_CURRICULUM_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(20.0, 20.0),  # Each patch: 20m x 20m (must be square, larger for nav)
    border_width=5.0,
    num_rows=6,     # 6 difficulty levels
    num_cols=10,    # 10 terrain type columns
    curriculum=True,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    sub_terrains={
        # --- 10% flat (always present for recovery) ---
        "flat": MeshPlaneTerrainCfg(
            proportion=0.10,
        ),
        # --- 15% random rough ---
        "random_rough": MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45,
            grid_height_range=(0.02, 0.12),  # Level 1: ~0.02m, Level 6: ~0.12m
            platform_width=3.0,
        ),
        # --- 15% boxes/obstacles ---
        "boxes": MeshRepeatedBoxesTerrainCfg(
            proportion=0.15,
            object_params_start=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=10,
                height=0.05,
                size=(0.5, 0.5),
            ),
            object_params_end=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=30,
                height=0.20,
                size=(0.4, 0.4),
            ),
            platform_width=3.0,
        ),
        # --- 12% stairs up ---
        "stairs_up": MeshPyramidStairsTerrainCfg(
            proportion=0.12,
            step_height_range=(0.03, 0.20),  # Level 1: 3cm, Level 6: 20cm
            step_width=0.30,
            platform_width=3.0,
        ),
        # --- 8% stairs down ---
        "stairs_down": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.08,
            step_height_range=(0.03, 0.20),
            step_width=0.30,
            platform_width=3.0,
        ),
        # --- 10% wave terrain ---
        "wave": HfWaveTerrainCfg(
            proportion=0.10,
            amplitude_range=(0.05, 0.15),  # Level 1: 5cm, Level 6: 15cm
            num_waves=3,
            border_width=0.25,
        ),
        # --- 10% discrete obstacles ---
        "discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
            proportion=0.10,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.3, 1.0),
            obstacle_height_range=(0.05, 0.20),
            num_obstacles=20,
            platform_width=2.0,
            border_width=0.25,
        ),
        # --- 15% polyhedral boulders (key terrain for nav decisions) ---
        "boulders": MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45,
            grid_height_range=(0.10, 0.80),  # Level 1: small 0.1m, Level 6: large 0.8m
            platform_width=3.0,
        ),
        # --- 5% friction plane (pure low-friction challenge, no geometry, no drag) ---
        "friction_plane": MeshPlaneTerrainCfg(
            proportion=0.05,
        ),
        # --- 5% vegetation plane (drag forces applied via VegetationDragReward) ---
        "vegetation_plane": MeshPlaneTerrainCfg(
            proportion=0.05,
        ),
    },
)


NAV_TERRAIN_IMPORTER_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=NAV_CURRICULUM_TERRAINS_CFG,
    debug_vis=False,
)
