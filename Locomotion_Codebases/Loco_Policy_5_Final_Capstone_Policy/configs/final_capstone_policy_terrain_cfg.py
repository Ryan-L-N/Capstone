"""Unified parkour-nav terrain curriculum.

Cribbed from Cheng 2024 Extreme Parkour's 5-column parkour split, adapted to
the 4 Spot test arenas (friction / grass / boulder / stairs). Each level-column
combination is one 8x8m patch; 10 difficulty levels x 20 columns = 200 patches.

Rule: difficulty ramps along rows; column determines terrain type. Promoted
by the standard `terrain_levels_vel` curriculum (distance vs commanded threshold).

Stair risers are explicitly ramped from 5cm (level 0) to 35cm (level 9) to
cover both 4_env_test stair zones (max 23cm) and Final World industrial
stairs (~18cm). Three different step_width variants train the policy on
narrow industrial (25cm), standard (30cm), and wide architectural (40cm)
run depths — Phase-7 (Apr 25) addition after teleop in Final World showed
the fixed-30cm run trained policy struggled with real-world stair geometry.
"""

import os
import sys

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

# Phase-10: register Loco_Policy_5_Final_Capstone_Policy custom sub-terrains
_LOCO5_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _LOCO5_ROOT not in sys.path:
    sys.path.insert(0, _LOCO5_ROOT)
from terrains import MeshOpenRiserStairsTerrainCfg  # noqa: E402, F401  # historical, retained for compatibility


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


# Phase-FW-Plus (Apr 27 PM): custom training for FW USD stair traversal +
# stair-zone-5 completion + descent capability. Resumes Phase-9 18500.
# - Adds open-riser STRAIGHT flights (no walls) — fixes the 3 FW failure
#   modes diagnosed via run_fw_stair_eval.py (foot-in-gap, body-wedge, side-drift)
# - Adds open-riser SWITCHBACK — for SM_Staircase_01 topology
# - Adds INVERTED PYRAMID stairs (Isaac Lab built-in) — for descent
# - Drops the Phase-10/10b open-riser PYRAMID (insufficient — pyramid
#   topology has implicit walls in 4 directions, doesn't teach the right skills)
_STAIR_RISER_RANGE = (0.05, 0.42)        # KEEP — solid pyramid stairs
# Phase-Final-B: cap open-riser at 0.20m (was 0.25m). Phase-Final collapsed
# at iter 1200 when curriculum promoted envs to level 3 open-riser at
# 0.18-0.25m risers — too much hard geometry, too fast. Cap at 0.20 keeps
# the policy near FW-realistic (~0.18m) without the upper cliff.
_OPEN_RISER_FW_RANGE = (0.15, 0.20)      # Phase-Final-B: 0.25 → 0.20
_OPEN_RISER_SWITCHBACK_RANGE = (0.15, 0.18)  # Phase-Final-B: 0.20 → 0.18
_BOULDER_RANGE = (0.03, 0.60)             # KEEP — Phase-8 calmed range


FINAL_CAPSTONE_POLICY_TERRAINS_CFG = TerrainGeneratorCfg(
    **_COMMON,
    sub_terrains={
        # =====================================================================
        # Stairs — Phase-7: 3 width variants per type to cover Final World
        # industrial stairs (25cm run) + standard (30cm) + wide architectural
        # (40cm). Total stair proportion 30% (same as Phase-4+), spread across
        # 6 sub-terrains so the policy sees diverse stair geometry.
        # =====================================================================
        # ===== Phase-FW-Plus stair stack (50% total) =====
        # Solid pyramid_stairs at 12% (was 17%) — preserved baseline,
        # 4% per width variant.
        "pyramid_stairs_narrow": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.04,
            step_height_range=_STAIR_RISER_RANGE,
            step_width=0.25,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_medium": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.08,  # Hail-Mary-4: +1% from open-riser removal
            step_height_range=_STAIR_RISER_RANGE,
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_wide": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.04,
            step_height_range=_STAIR_RISER_RANGE,
            step_width=0.40,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # Solid hf_stairs at 12% (was 18%) — preserved baseline.
        "hf_stairs_narrow": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.04,
            step_height_range=_STAIR_RISER_RANGE,
            step_width=0.25,
            border_width=0.25,
        ),
        "hf_stairs_medium": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.07,  # Phase-Final-B: 0.04 → 0.07 (+3% from open-riser cut)
            step_height_range=_STAIR_RISER_RANGE,
            step_width=0.30,
            border_width=0.25,
        ),
        "hf_stairs_wide": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.04,
            step_height_range=_STAIR_RISER_RANGE,
            step_width=0.40,
            border_width=0.25,
        ),
        # ===== Phase-FW-Plus NEW: descent training (8%) =====
        # Inverted pyramid — robot spawns on the central plateau and
        # descends OUTWARD. Same step_height_range as solid pyramid.
        # Curriculum still rewards distance from origin, so promotion
        # happens when robots successfully descend.
        "inverted_pyramid_stairs_medium": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.04,
            step_height_range=_STAIR_RISER_RANGE,
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "inverted_pyramid_stairs_narrow": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.04,
            step_height_range=_STAIR_RISER_RANGE,
            step_width=0.25,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # ===== Hail-Mary-4: open-riser sub-terrains REMOVED =====
        # parkour_hailmary3 chronically oscillated (15+ vf_loss spikes by iter
        # 122) on these because the no-side-walls topology produces bimodal
        # returns: robot either tracks the tread strip (high reward) or drifts
        # off the open side (zero reward). Critic can't fit such high-variance
        # V_target at any LR. Removed entirely — 9% reallocated to flat (6%),
        # pyramid_stairs_medium (1%), flat_clutter (2%). Standard pyramid_stairs
        # already have solid risers, which is what Colby's modified FW USDs
        # will have once riser-baked, so this terrain class is sufficient.

        # Columns 4-7: boulders / discrete obstacles. Phase-10b: bump boulder
        # proportion 5% → 7% (steal 2% from pyramid_stairs_medium). Phase-9 stole
        # from slopes for stairs and shifted gait away from rough-rock placements,
        # causing boulder z3 FLIP regression. More boulder reps without changing
        # difficulty re-balances the gait without undoing Phase-8 boulder calming.
        "boulders": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.07,  # Phase-10b: 0.05 → 0.07
            obstacle_height_mode="choice",
            obstacle_height_range=_BOULDER_RANGE,
            obstacle_width_range=(0.20, 0.80),
            num_obstacles=40,
            platform_width=2.0,
        ),
        "random_boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.05,
            grid_width=0.45,
            grid_height_range=(0.03, 0.18),
            platform_width=2.0,
        ),

        # Columns 8-11: slopes — Phase-9: slope_up 10% → 5% (steal for stairs)
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05,  # Phase-9: 0.10 → 0.05 (donates to stairs)
            slope_range=(0.0, 0.50),  # Phase-3: 0.40→0.50 rad (~29°)
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.13,  # Phase-Final-B: 0.10 → 0.13 (+3% from open-riser cut)
            noise_range=(0.02, 0.15),
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
            proportion=0.20,  # Hail-Mary-4: +2% from open-riser removal
            obstacle_height_mode="choice",
            obstacle_height_range=(0.05, 0.45),
            obstacle_width_range=(0.20, 0.80),
            num_obstacles=45,
            platform_width=2.0,
        ),

        # Phase-8: flat 5% → 15% (TRIPLE) to recover Cole-style flat-ground
        # stability that Phase-7 over-shifted away from. stepping_stones kept
        # at 5%. Net: +10% flat (from 5% boulder + 5% random_box reductions).
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.05,
            stone_height_max=0.08,
            stone_width_range=(0.25, 0.60),
            stone_distance_range=(0.05, 0.25),
            platform_width=2.0,
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.06,  # Hail-Mary-4: 0.0 → 0.06 (low-variance anchor for stable
                               # critic warmup; absorbs 6% of the 9% open-riser cut).
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
