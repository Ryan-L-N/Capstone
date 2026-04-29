"""Custom terrain generator for 100hr multi-terrain robust locomotion training.

12 terrain types across 3 categories, optimized for training a policy that
handles rough terrain, stairs (up AND down), boulders, low-friction surfaces,
and unstructured terrain.

Grid: 10 rows (difficulty progression) x 40 cols (terrain variety) = 400 patches.
Each patch is 8m x 8m.

Changes from ROUGH_TERRAINS_CFG (6 types, 10x20):
  - 12 terrain types (2x diversity)
  - 40 columns (2x terrain variety per difficulty level)
  - Added: wave terrain, stepping stones, gaps, discrete obstacles
  - Wider difficulty ranges (steeper stairs, rougher terrain)
  - Inverted stairs for descending practice

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
        # Wider height range than default (0.05-0.25m vs 0.05-0.2m)
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

        # Random rough — general uneven ground, wider noise than default
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10,
            noise_range=(0.02, 0.15),
            noise_step=0.02,
            border_width=0.25,
        ),

        # Uphill slopes — steeper than default (0.0-0.5 vs 0.0-0.4)
        "hf_pyramid_slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.075,
            slope_range=(0.0, 0.5),
            platform_width=2.0,
            border_width=0.25,
        ),

        # Downhill slopes — same range, inverted
        "hf_pyramid_slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.075,
            slope_range=(0.0, 0.5),
            platform_width=2.0,
            border_width=0.25,
        ),

        # Wave terrain — undulating ground (simulates natural terrain contours)
        "wave_terrain": terrain_gen.HfWaveTerrainCfg(
            proportion=0.05,
            amplitude_range=(0.05, 0.2),
            num_waves=3,
            border_width=0.25,
        ),

        # ─── FRICTION PLANE (5%) ──────────────────────────────────────
        # Perfectly flat surface where LOW FRICTION is the sole challenge.
        # No vegetation drag is applied here — the difficulty comes
        # entirely from the per-env friction randomization (mu down to
        # 0.05 = oil on polished steel).
        #
        # Teaches the policy to balance and walk on slippery surfaces
        # without the confounding factor of drag resistance.
        "friction_plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.05,
        ),

        # ─── VEGETATION PLANE (5%) ───────────────────────────────────
        # Perfectly flat surface where VEGETATION DRAG is the sole
        # challenge. VegetationDragReward always applies drag > 0 here
        # (c = 0.5 to 20.0 N*s/m), while friction stays at whatever
        # the global randomization assigns.
        #
        # Teaches the policy to push through grass/mud/brush resistance
        # without the confounding factor of terrain geometry.
        "vegetation_plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.05,
        ),

        # =================================================================
        # CATEGORY C: Compound Challenges (25%)
        # =================================================================

        # Heightfield stairs — coarser, noisier stairs (debris on steps)
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
"""Robust multi-terrain configuration: 40% geometric + 35% surface (5% friction + 5% vegetation) + 25% compound."""
