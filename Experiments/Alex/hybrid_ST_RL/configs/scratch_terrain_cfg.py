"""Attempt 5: Terrain curriculum for from-scratch training.

7 terrain types focused on the user's requirements:
  - Flat (20%) — true flat for learning basic locomotion
  - Random rough / uneven (20%) — general uneven ground
  - Boxes / boulders (15%) — rubble-like obstacles
  - Stairs up (15%) — ascending stairs
  - Stairs down (15%) — descending stairs
  - Friction plane (10%) — flat + low friction challenge
  - Vegetation plane (5%) — flat + drag challenge

Grid: 10 rows (difficulty 0-9) x 30 cols (variety) = 300 patches, 8m x 8m.
Curriculum enabled — robots auto-promote/demote based on velocity tracking.
All robots start at level 0 where non-flat terrains have minimal features
(5cm steps, 2cm roughness = effectively flat).

Created for AI2C Tech Capstone — hybrid_ST_RL Attempt 5, February 2026
"""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

SCRATCH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=30,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # =================================================================
        # TRUE FLAT (20%) — learn to walk here first
        # =================================================================
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.20,
        ),

        # =================================================================
        # UNEVEN GROUND (20%) — general rough terrain
        # =================================================================
        # At difficulty 0: 0.02m noise (barely noticeable)
        # At difficulty 9: 0.15m noise (challenging uneven ground)
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.02, 0.15),
            noise_step=0.02,
            border_width=0.25,
        ),

        # =================================================================
        # BOULDERS (15%) — random grid boxes as boulder proxy
        # =================================================================
        # At difficulty 0: 0.05m height (tiny bumps)
        # At difficulty 9: 0.25m height (serious obstacles)
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45,
            grid_height_range=(0.05, 0.25),
            platform_width=2.0,
        ),

        # =================================================================
        # STAIRS UP (15%) — ascending stairs
        # =================================================================
        # At difficulty 0: 0.05m steps (gentle ramp)
        # At difficulty 9: 0.25m steps (extreme stairs)
        "pyramid_stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),

        # =================================================================
        # STAIRS DOWN (15%) — descending stairs
        # =================================================================
        # Same height range as stairs up, inverted
        "pyramid_stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),

        # =================================================================
        # FRICTION PLANE (10%) — flat surface, low-friction challenge
        # =================================================================
        # No terrain geometry — difficulty comes from per-env friction
        # randomization (mu as low as 0.05 = oil on polished steel).
        "friction_plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.10,
        ),

        # =================================================================
        # VEGETATION PLANE (5%) — flat surface, drag challenge
        # =================================================================
        # VegetationDragReward applies drag > 0 here (c = 0.5 to 20.0 N*s/m).
        # Teaches the policy to push through resistance.
        "vegetation_plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.05,
        ),
    },
)
