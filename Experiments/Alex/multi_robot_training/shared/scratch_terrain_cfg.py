"""7-terrain warmup/testing configuration.

Simpler terrain set for warmup training and quick testing.
7 terrain types focused on core locomotion challenges:
  - Flat (20%), Random rough (20%), Boxes (15%), Stairs up (15%),
    Stairs down (15%), Friction plane (10%), Vegetation plane (5%)

Grid: 10 rows (difficulty) x 30 cols (variety) = 300 patches, 8m x 8m.
Curriculum enabled — robots auto-promote/demote based on velocity tracking.

Source: vision60_training/configs/scratch_terrain_cfg.py
Created for AI2C Tech Capstone — MS for Autonomy, February 2026
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
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.20,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.02, 0.15),
            noise_step=0.02,
            border_width=0.25,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45,
            grid_height_range=(0.05, 0.25),
            platform_width=2.0,
        ),
        "pyramid_stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "friction_plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.10,
        ),
        "vegetation_plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.05,
        ),
    },
)
