"""Custom terrain configurations for 6 terrain-specialist experts + distillation.

Each expert gets 80% of its specialty terrain + 20% flat baseline.
Grid: 10 rows (difficulty) x 20 cols = 200 patches, 8m x 8m each.
Curriculum enabled for progressive difficulty.

Created for AI2C Tech Capstone — MS for Autonomy, March 2026
"""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


# Common terrain generator settings (shared across all configs)
_COMMON = dict(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
)


# =============================================================================
# Expert 1: FRICTION — 80% friction planes + 20% flat
# =============================================================================

FRICTION_TERRAINS_CFG = TerrainGeneratorCfg(
    **_COMMON,
    sub_terrains={
        "friction_plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.80,
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.20,
        ),
    },
)


# =============================================================================
# Expert 2: STAIRS UP — 40% pyramid_stairs + 40% hf_stairs + 20% flat
# =============================================================================

STAIRS_UP_TERRAINS_CFG = TerrainGeneratorCfg(
    **_COMMON,
    sub_terrains={
        "pyramid_stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.40,
            step_height_range=(0.03, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_stairs_up": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.40,
            step_height_range=(0.03, 0.20),
            step_width=0.3,
            border_width=0.25,
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.20,
        ),
    },
)


# =============================================================================
# Expert 3: STAIRS DOWN — 80% descending stairs + 20% flat
# =============================================================================

STAIRS_DOWN_TERRAINS_CFG = TerrainGeneratorCfg(
    **_COMMON,
    sub_terrains={
        "pyramid_stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.50,
            step_height_range=(0.03, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # Heightfield variant for diversity
        "hf_stairs_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.30,
            slope_range=(0.2, 0.5),
            platform_width=2.0,
            border_width=0.25,
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.20,
        ),
    },
)


# =============================================================================
# Expert 4: BOULDERS — 40% boxes + 20% discrete + 20% repeated + 20% flat
# =============================================================================

BOULDER_TERRAINS_CFG = TerrainGeneratorCfg(
    **_COMMON,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.35,
            grid_width=0.45,
            grid_height_range=(0.05, 0.25),
            platform_width=2.0,
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.25,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.25, 0.75),
            obstacle_height_range=(0.05, 0.30),
            num_obstacles=40,
            platform_width=2.0,
            border_width=0.25,
        ),
        "repeated_boxes": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            proportion=0.20,
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
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.20,
        ),
    },
)


# =============================================================================
# Expert 5: SLOPES — 35% up + 35% down + 10% wave + 20% flat
# =============================================================================

SLOPES_TERRAINS_CFG = TerrainGeneratorCfg(
    **_COMMON,
    sub_terrains={
        "hf_pyramid_slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.35,
            slope_range=(0.0, 0.5),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.35,
            slope_range=(0.0, 0.5),
            platform_width=2.0,
            border_width=0.25,
        ),
        "wave_terrain": terrain_gen.HfWaveTerrainCfg(
            proportion=0.10,
            amplitude_range=(0.05, 0.2),
            num_waves=3,
            border_width=0.25,
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.20,
        ),
    },
)


# =============================================================================
# Expert 6: MIXED ROUGH — 40% random_rough + 40% stepping_stones + 20% flat
# =============================================================================

MIXED_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    **_COMMON,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.40,
            noise_range=(0.02, 0.15),
            noise_step=0.02,
            border_width=0.25,
        ),
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.40,
            stone_height_max=0.15,
            stone_width_range=(0.25, 0.5),
            stone_distance_range=(0.1, 0.4),
            border_width=0.25,
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.20,
        ),
    },
)


# =============================================================================
# DISTILLATION — Balanced all-terrain mix for generalist student
# =============================================================================

DISTILLATION_TERRAINS_CFG = TerrainGeneratorCfg(
    **_COMMON,
    num_cols=40,  # More columns for terrain variety during distillation
    sub_terrains={
        # Stairs (20%)
        "pyramid_stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.03, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.03, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),

        # Boulders/obstacles (15%)
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.10,
            grid_width=0.45,
            grid_height_range=(0.05, 0.25),
            platform_width=2.0,
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.05,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.25, 0.75),
            obstacle_height_range=(0.05, 0.30),
            num_obstacles=40,
            platform_width=2.0,
            border_width=0.25,
        ),

        # Slopes (15%)
        "hf_pyramid_slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.075,
            slope_range=(0.0, 0.5),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.075,
            slope_range=(0.0, 0.5),
            platform_width=2.0,
            border_width=0.25,
        ),

        # Rough/stepping (15%)
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10,
            noise_range=(0.02, 0.15),
            noise_step=0.02,
            border_width=0.25,
        ),
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.05,
            stone_height_max=0.15,
            stone_width_range=(0.25, 0.5),
            stone_distance_range=(0.1, 0.4),
            border_width=0.25,
        ),

        # Surface/wave (10%)
        "friction_plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.05,
        ),
        "wave_terrain": terrain_gen.HfWaveTerrainCfg(
            proportion=0.05,
            amplitude_range=(0.05, 0.2),
            num_waves=3,
            border_width=0.25,
        ),

        # Flat baseline (10%)
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.10,
        ),

        # HF stairs for variety (10%)
        "hf_stairs_up": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.03, 0.20),
            step_width=0.3,
            border_width=0.25,
        ),
    },
)
