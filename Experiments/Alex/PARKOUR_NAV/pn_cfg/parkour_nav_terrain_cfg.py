"""Unified parkour-nav terrain curriculum.

Cribbed from Cheng 2024 Extreme Parkour's 5-column parkour split, adapted to
the 4 Spot test arenas (friction / grass / boulder / stairs). Each level-column
combination is one 8x8m patch; 10 difficulty levels x 20 columns = 200 patches.

Rule: difficulty ramps along rows; column determines terrain type. Promoted
by the standard `terrain_levels_vel` curriculum (distance vs commanded threshold).

Stair risers are explicitly ramped from 5cm (level 0) to 23cm (level 9) to
hit the 4_env_test stairs arena zone-5 riser target (23cm).
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


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


# Phase-3 hardening (Apr 24): extended stair ceiling past 4_env zone-5 target
# (23cm) to 30cm so high-curriculum patches force sub-zone-5 generalization.
# Boulders also bumped 50→60 cm for harder rock-field zone-3 exposure — iter
# 6000 flipped at boulder zone 3 so we want denser training near that regime.
_STAIR_RISER_RANGE = (0.03, 0.30)
_BOULDER_RANGE = (0.03, 0.60)


PARKOUR_NAV_TERRAINS_CFG = TerrainGeneratorCfg(
    **_COMMON,
    sub_terrains={
        # Columns 0-3: stairs up (20% of patches)
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=_STAIR_RISER_RANGE,
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_stairs": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=_STAIR_RISER_RANGE,
            step_width=0.30,
            border_width=0.25,
        ),

        # Columns 4-7: boulders / discrete obstacles (20%)
        "boulders": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.10,
            obstacle_height_mode="choice",
            obstacle_height_range=_BOULDER_RANGE,
            obstacle_width_range=(0.20, 0.80),
            num_obstacles=40,  # Phase-3: 30→40 denser rock fields
            platform_width=2.0,
        ),
        "random_boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.10,
            grid_width=0.45,
            grid_height_range=(0.03, 0.18),
            platform_width=2.0,
        ),

        # Columns 8-11: slopes (20%)
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.0, 0.50),  # Phase-3: 0.40→0.50 rad (~29°)
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10,
            noise_range=(0.02, 0.15),  # Phase-3: 0.10→0.15 rougher noise
            noise_step=0.02,
            border_width=0.25,
        ),

        # Columns 12-15: flat with obstacle scatter (Cole-style nav) (20%)
        # Cole's rich arena has ~46 obstacles at 0.25/0.25/0.0 density across
        # 7 shape types. HfDiscreteObstaclesTerrainCfg rasterizes boxes into
        # the height field — functionally equivalent for a policy learning to
        # navigate around them. Width range 0.2-0.8m spans Cole's box/cone/
        # cylinder/pillar size envelope; heights ramp from pebbles to waist.
        "flat_clutter": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.20,
            obstacle_height_mode="choice",
            obstacle_height_range=(0.05, 0.45),
            obstacle_width_range=(0.20, 0.80),
            num_obstacles=45,  # Phase-3: 35→45 for denser Cole-style clutter
            platform_width=2.0,
        ),

        # Columns 16-19: mixed / gaps / stepping (20%)
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.10,
            stone_height_max=0.08,
            stone_width_range=(0.25, 0.60),
            stone_distance_range=(0.05, 0.25),  # Phase-3: 0.15→0.25 wider gaps
            platform_width=2.0,
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.10,
        ),
    },
)


# NOTE: the "scatter on every patch" design goal (Cole-compatibility twist)
# would require a post-process that drops obstacles INTO stair/boulder/slope
# patches. Isaac Lab's terrain generator composes sub-terrains side-by-side
# rather than layered, so a true post-process needs either a custom sub-terrain
# function that accepts a "base geometry + add obstacles" signature, or runtime
# spawning via InteractiveScene props. Deferred to Phase 2/3 — HfDiscreteObstacles
# in `flat_clutter` + natural clutter in `boulders` + `random_boxes` already
# gives the policy ~40% clutter exposure across the curriculum, which is the
# parkour-paper operating point. Revisit if teacher stalls on Cole rich max.
