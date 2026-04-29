"""Open-side STRAIGHT-FLIGHT stair sub-terrain — Final Capstone Policy build.

Single-direction stair flight with NO side walls but WITH solid riser faces
between treads. Matches the geometry Colby's modified SM_Staircase_*.usd
files will produce after riser-baking (per FW_Stairs_Riser_Project/
RISER_INTEGRATION_PLAN.md). Training distribution now matches the
deployment distribution from iter 0.

Spot approaches from the -X apron, climbs across the flight (solid tread
+ solid riser per step), and arrives on the +X top platform.

Compared to MeshOpenRiserStairsTerrainCfg (pyramid topology):
  - Pyramid had 4-way symmetric stairs converging on a center platform —
    implicit walls in 4 directions kept the policy from learning side-drift
    recovery and let it use neighbor-rings as cheats.
  - Straight flight has open sides — Spot must hold lateral position with
    no wall reference. Models the dominant FW failure mode (lateral drift
    off the stair).
  - Pyramid had wide treads (full ring extent) — body never had to span a
    gap. Straight flight has tread strips of width=step_width along X.
    Solid risers now fill the front face of each tread so the foot can
    brace against them on the way up (the original "tread only" version
    had a foot-in-gap failure mode that mirrored the broken FW USDs).

Layout (in patch-local coords, patch is size = (cfg.size[0], cfg.size[1])):

   Y
   ^
   |
   +-------------------------------------+ <- patch +Y edge
   |  apron  |  treads ascending +X      | top platform
   |  (flat) | T1  T2  T3  T4  ...   Tn  |  (flat at top z)
   |   (0)   |     (z increases)         |
   +-------------------------------------+ <- patch -Y edge
   ^         ^                           ^
   X=0      apron_end                  patch +X edge
            (X = approach_apron_len)    (X = patch.size[0])

Each tread Tk is a thin box (step_width × full Y × tread_thickness)
centered at (apron_end + (k + 0.5)*step_width, patch_center_y, k*step_height + th/2).

Each riser Rk is a thin vertical box (riser_thickness × full Y × step_height)
at the leading -X edge of tread Tk: X = apron_end + k*step_width.
Riser Rk spans z = [k*step_height, (k+1)*step_height], filling the gap below
the next tread top so an ascending foot has a vertical surface to brace
against. The space BEHIND the riser (between successive risers, below the
tread plane) is left open — this matches the FW USD "open back" geometry
once Colby bakes risers in.

Top platform is a solid box at the +X end so Spot has somewhere to stand
when ascent completes.
"""
from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass


def open_riser_straight_terrain(
    difficulty: float, cfg: "MeshOpenRiserStraightFlightCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a single-direction open-riser stair flight."""
    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    # tread thickness: clamp so gap is at least 60% of riser, never bigger than 8cm
    tread_thickness = min(cfg.tread_thickness, 0.4 * step_height, 0.08)
    tread_thickness = max(tread_thickness, 0.02)

    patch_x = cfg.size[0]
    patch_y = cfg.size[1]
    apron_len = cfg.approach_apron_len_m
    top_platform_len = cfg.top_platform_len_m

    flight_len = patch_x - apron_len - top_platform_len
    if flight_len <= 0.5:
        # patch too small for the flight — fall back to flat
        plane = trimesh.creation.box(
            (patch_x, patch_y, 0.04),
            trimesh.transformations.translation_matrix((patch_x / 2.0, patch_y / 2.0, -0.02)),
        )
        return [plane], np.array([apron_len / 2.0, patch_y / 2.0, 0.0])

    num_steps = max(1, int(flight_len // cfg.step_width))
    actual_flight_len = num_steps * cfg.step_width
    apron_end_x = apron_len  # treads start at this X

    meshes: list[trimesh.Trimesh] = []

    # Approach apron: solid box at z = -tread_thickness/2 .. 0 (top at z=0)
    apron = trimesh.creation.box(
        (apron_len, patch_y, 0.04),
        trimesh.transformations.translation_matrix((apron_len / 2.0, patch_y / 2.0, -0.02)),
    )
    meshes.append(apron)

    # Treads (thin horizontal slabs, no side walls) + solid risers
    riser_thickness = cfg.riser_thickness
    riser_overlap = 0.005  # tiny vertical overlap to avoid Z-fight with tread
    for k in range(num_steps):
        z_top = (k + 1) * step_height
        z_center = z_top - tread_thickness / 2.0
        x_center = apron_end_x + (k + 0.5) * cfg.step_width
        tread = trimesh.creation.box(
            (cfg.step_width, patch_y, tread_thickness),
            trimesh.transformations.translation_matrix((x_center, patch_y / 2.0, z_center)),
        )
        meshes.append(tread)

        # Solid riser face on the -X edge of tread k.
        # Spans z = [k*step_height, (k+1)*step_height], i.e. from previous
        # tread top up to current tread top. For k=0 this is from apron top
        # (z=0) to tread 0 top — the very first riser visible to a robot
        # walking onto the flight from the apron.
        riser_z_low = k * step_height
        riser_z_high = z_top
        riser_z_center = (riser_z_low + riser_z_high) / 2.0
        riser_z_extent = riser_z_high - riser_z_low + riser_overlap
        riser_x_center = apron_end_x + k * cfg.step_width
        riser = trimesh.creation.box(
            (riser_thickness, patch_y, riser_z_extent),
            trimesh.transformations.translation_matrix(
                (riser_x_center, patch_y / 2.0, riser_z_center)
            ),
        )
        meshes.append(riser)

    # Top platform — solid pillar from -tread_thickness/2 up to top z
    top_z = num_steps * step_height
    top_platform_x_start = apron_end_x + actual_flight_len
    top_platform_x_center = top_platform_x_start + top_platform_len / 2.0
    top_box = trimesh.creation.box(
        (top_platform_len, patch_y, top_z + 0.04),
        trimesh.transformations.translation_matrix(
            (top_platform_x_center, patch_y / 2.0, (top_z + 0.04) / 2.0 - 0.02)
        ),
    )
    meshes.append(top_box)

    # Origin = spawn point on the apron, facing the stair (+X).
    # Curriculum measures distance from origin, so promotion happens when
    # robots traverse the flight and reach the top.
    origin = np.array([apron_len / 2.0, patch_y / 2.0, 0.0])
    return meshes, origin


@configclass
class MeshOpenRiserStraightFlightCfg(SubTerrainBaseCfg):
    """Single-direction stair flight, solid risers + open sides.

    Models the FW SM_Staircase_02 / SM_StaircaseHalf_02 topology AFTER
    Colby's riser-baking — solid tread + solid riser per step, but no
    side walls. The lateral-drift failure mode is preserved (open sides);
    the foot-in-gap and body-wedge failure modes are eliminated by the
    solid riser faces. Final Capstone Policy from-scratch trains the policy on this
    geometry from iter 0.
    """

    function = open_riser_straight_terrain

    step_height_range: tuple[float, float] = MISSING
    """[min, max] riser height (m). Final Capstone Policy uses (0.15, 0.20) — FW-realistic."""

    step_width: float = MISSING
    """Run depth per tread (m). Final Capstone Policy uses 0.25-0.30 m to match FW industrial dim."""

    approach_apron_len_m: float = 1.5
    """Length of the flat approach patch at the -X end (m). Spot spawns here."""

    top_platform_len_m: float = 1.0
    """Length of the flat platform at the +X end (m). Goal of the ascent."""

    tread_thickness: float = 0.04
    """Vertical thickness of each tread slab (m). Auto-clamped to 40% of step_height."""

    riser_thickness: float = 0.02
    """Thickness of each solid riser face along X (m). 2 cm matches FW USD."""
