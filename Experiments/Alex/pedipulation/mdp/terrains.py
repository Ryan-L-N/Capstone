"""Staircase-heavy terrain configuration for pedipulation training.

30% flat, 30% stairs up, 15% stairs down, 10% slopes, 10% rough, 5% HF stairs.
Designed for training a robot to push objects on staircases while balancing.

Grid: 10 rows (difficulty) x 30 cols = 300 patches, 8m x 8m each.
Curriculum enabled for progressive difficulty.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


PEDI_STAIRCASE_TERRAINS_CFG = TerrainGeneratorCfg(
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
        # 30% flat — safe zone for learning manipulation basics
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.30,
        ),

        # 30% ascending stairs — primary challenge
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),

        # 15% descending stairs
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),

        # 10% slopes — ramp surfaces
        "slopes": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.0, 0.5),
            platform_width=2.0,
            border_width=0.25,
        ),

        # 10% random rough ground
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10,
            noise_range=(0.02, 0.15),
            noise_step=0.02,
            border_width=0.25,
        ),

        # 5% heightfield stairs — coarser, noisier stairs
        "hf_stairs": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            border_width=0.25,
        ),
    },
)
