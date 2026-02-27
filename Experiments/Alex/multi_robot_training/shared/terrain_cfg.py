"""Robust 12-terrain configuration for multi-robot training.

12 terrain types across 3 categories, optimized for training policies that
handle rough terrain, stairs (up AND down), boulders, low-friction surfaces,
and unstructured terrain.

Grid: 10 rows (difficulty progression) x 40 cols (terrain variety) = 400 patches.
Each patch is 8m x 8m.

Terrain breakdown:
  Category A — Geometric Obstacles (40%):
    pyramid_stairs_up (10%), pyramid_stairs_down (10%), boxes (10%),
    stepping_stones (5%), gaps (5%)
  Category B — Surface Variation (35%):
    random_rough (10%), hf_pyramid_slope_up (7.5%), hf_pyramid_slope_down (7.5%),
    wave_terrain (5%), friction_plane (5%), vegetation_plane (5%)
  Category C — Compound Challenges (25%):
    hf_stairs_up (10%), discrete_obstacles (5%), repeated_boxes (5%)

Source: 100hr_env_run/configs/terrain_cfg.py
Created for AI2C Tech Capstone — MS for Autonomy, February 2026
"""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

ROBUST_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=40,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # =================================================================
        # CATEGORY A: Geometric Obstacles (40%)
        # =================================================================

        # Ascending stairs — 0.05m (access ramp) to 0.25m (extreme challenge)
        "pyramid_stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),

        # Descending stairs — same range, inverted
        "pyramid_stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),

        # Random grid boxes — unstructured obstacles (rubble/boulder proxy)
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.10,
            grid_width=0.45,
            grid_height_range=(0.05, 0.25),
            platform_width=2.0,
        ),

        # Stepping stones — precise foot placement training
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.05,
            stone_height_max=0.15,
            stone_width_range=(0.25, 0.5),
            stone_distance_range=(0.1, 0.4),
            border_width=0.25,
        ),

        # Gaps — stride over or jump across
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.05,
            gap_width_range=(0.1, 0.5),
            platform_width=2.0,
        ),

        # =================================================================
        # CATEGORY B: Surface Variation (35%)
        # =================================================================

        # Random rough — general uneven ground
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10,
            noise_range=(0.02, 0.15),
            noise_step=0.02,
            border_width=0.25,
        ),

        # Uphill slopes
        "hf_pyramid_slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.075,
            slope_range=(0.0, 0.5),
            platform_width=2.0,
            border_width=0.25,
        ),

        # Downhill slopes
        "hf_pyramid_slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.075,
            slope_range=(0.0, 0.5),
            platform_width=2.0,
            border_width=0.25,
        ),

        # Wave terrain — undulating ground
        "wave_terrain": terrain_gen.HfWaveTerrainCfg(
            proportion=0.05,
            amplitude_range=(0.05, 0.2),
            num_waves=3,
            border_width=0.25,
        ),

        # Friction plane — pure low-friction challenge (no geometry, no drag)
        "friction_plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.05,
        ),

        # Vegetation plane — pure drag challenge (no geometry)
        "vegetation_plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.05,
        ),

        # =================================================================
        # CATEGORY C: Compound Challenges (25%)
        # =================================================================

        # Heightfield stairs — coarser, noisier stairs
        "hf_stairs_up": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            border_width=0.25,
        ),

        # Discrete obstacles — scattered blocks of varying heights
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.05,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.25, 0.75),
            obstacle_height_range=(0.05, 0.30),
            num_obstacles=40,
            platform_width=2.0,
            border_width=0.25,
        ),

        # Repeated boxes — regular pattern of obstacles
        "repeated_boxes": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            proportion=0.05,
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=20,
                height=0.05,
                size=(0.3, 0.3),
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=40,
                height=0.20,
                size=(0.5, 0.5),
            ),
            platform_width=2.0,
        ),
    },
)
