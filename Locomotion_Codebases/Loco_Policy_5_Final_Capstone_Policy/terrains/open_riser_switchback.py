"""Open-side SWITCHBACK stair sub-terrain — Final Capstone Policy build.

Two stair flights connected by a flat landing, second flight rotated 180°
from first. Models the SM_Staircase_01 topology AFTER Colby's riser-baking
— solid tread + solid riser per step on each flight, no side walls
between flights. Lateral-drift failure preserved; foot-in-gap and
body-wedge failures eliminated by solid risers.

Layout (patch in patch-local coords, size = (cfg.size[0], cfg.size[1])):

    Y
    ^
    |
    +-------------------------------------+ <- patch +Y
    | top  | flight 2 (-X heading) treads |  flat   |
    +------+------------------------------+         |
    |                                     | landing |   <- y_landing band
    +------+------------------------------+         |
    | apron| flight 1 (+X heading) treads |  flat   |
    +------+------------------------------+ <- patch -Y
    ^      ^                              ^         ^
    X=0   apron_len                  flight_end_x  patch.size[0]

Flight 1 ascends from z=0 to z=mid_h heading +X.
Landing is a flat solid platform at z=mid_h spanning the +X side full-Y.
Flight 2 ascends from z=mid_h to z=top_h heading -X (opposite).
Top platform is at the -X side, +Y half, at z=top_h.
"""
from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass


def open_riser_switchback_terrain(
    difficulty: float, cfg: "MeshOpenRiserSwitchbackCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate two-flight switchback with open risers and a landing."""
    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    tread_thickness = min(cfg.tread_thickness, 0.4 * step_height, 0.08)
    tread_thickness = max(tread_thickness, 0.02)

    patch_x = cfg.size[0]
    patch_y = cfg.size[1]
    apron_len = cfg.approach_apron_len_m
    top_platform_len = cfg.top_platform_len_m
    landing_len = cfg.landing_len_m
    flight_y_strip = (patch_y - cfg.flight_separation_m) / 2.0  # each flight occupies this much Y

    if flight_y_strip <= 0.5:
        # patch too narrow for two flights — fallback flat
        plane = trimesh.creation.box(
            (patch_x, patch_y, 0.04),
            trimesh.transformations.translation_matrix((patch_x / 2.0, patch_y / 2.0, -0.02)),
        )
        return [plane], np.array([apron_len / 2.0, flight_y_strip / 2.0, 0.0])

    flight_x_len = patch_x - apron_len - landing_len
    num_steps = max(1, int(flight_x_len // cfg.step_width))
    actual_flight_x = num_steps * cfg.step_width
    mid_h = num_steps * step_height
    top_h = 2 * num_steps * step_height

    apron_end_x = apron_len
    flight_end_x = apron_end_x + actual_flight_x
    landing_start_x = flight_end_x

    meshes: list[trimesh.Trimesh] = []

    # Apron at -Y half (where flight 1 starts)
    apron_y_center = flight_y_strip / 2.0
    apron = trimesh.creation.box(
        (apron_len, flight_y_strip, 0.04),
        trimesh.transformations.translation_matrix((apron_len / 2.0, apron_y_center, -0.02)),
    )
    meshes.append(apron)

    # Flight 1 treads + solid risers — heading +X, at -Y strip
    riser_thickness = cfg.riser_thickness
    riser_overlap = 0.005
    for k in range(num_steps):
        z_top = (k + 1) * step_height
        z_center = z_top - tread_thickness / 2.0
        x_center = apron_end_x + (k + 0.5) * cfg.step_width
        tread = trimesh.creation.box(
            (cfg.step_width, flight_y_strip, tread_thickness),
            trimesh.transformations.translation_matrix(
                (x_center, apron_y_center, z_center)
            ),
        )
        meshes.append(tread)

        # Riser at -X edge of flight-1 tread k (Spot ascends +X, so the
        # riser face Spot brushes against on the way up sits on the -X side
        # of each tread).
        riser_z_low = k * step_height
        riser_z_high = z_top
        riser_z_center = (riser_z_low + riser_z_high) / 2.0
        riser_z_extent = riser_z_high - riser_z_low + riser_overlap
        riser_x_center = apron_end_x + k * cfg.step_width
        riser = trimesh.creation.box(
            (riser_thickness, flight_y_strip, riser_z_extent),
            trimesh.transformations.translation_matrix(
                (riser_x_center, apron_y_center, riser_z_center)
            ),
        )
        meshes.append(riser)

    # Landing — solid platform at z=mid_h, spans full Y, +X side, length landing_len
    landing = trimesh.creation.box(
        (landing_len, patch_y, mid_h + 0.04),
        trimesh.transformations.translation_matrix(
            (landing_start_x + landing_len / 2.0, patch_y / 2.0, (mid_h + 0.04) / 2.0 - 0.02)
        ),
    )
    meshes.append(landing)

    # Flight 2 treads + solid risers — heading -X, at +Y strip
    flight2_y_center = patch_y - flight_y_strip / 2.0
    for k in range(num_steps):
        # k=0 is closest to landing; tread top z = mid_h + (k+1)*step_height
        z_top = mid_h + (k + 1) * step_height
        z_center = z_top - tread_thickness / 2.0
        # Walk -X from landing_start_x toward apron_end_x
        x_center = flight_end_x - (k + 0.5) * cfg.step_width
        tread = trimesh.creation.box(
            (cfg.step_width, flight_y_strip, tread_thickness),
            trimesh.transformations.translation_matrix(
                (x_center, flight2_y_center, z_center)
            ),
        )
        meshes.append(tread)

        # Riser at +X edge of flight-2 tread k. Spot descends -X on this
        # flight (after the landing turn), so the riser Spot's foot brushes
        # against on the way up sits on the +X side of each tread (the side
        # facing the previous tread, which is one tread closer to the
        # landing).
        riser_z_low = mid_h + k * step_height
        riser_z_high = z_top
        riser_z_center = (riser_z_low + riser_z_high) / 2.0
        riser_z_extent = riser_z_high - riser_z_low + riser_overlap
        riser_x_center = flight_end_x - k * cfg.step_width
        riser = trimesh.creation.box(
            (riser_thickness, flight_y_strip, riser_z_extent),
            trimesh.transformations.translation_matrix(
                (riser_x_center, flight2_y_center, riser_z_center)
            ),
        )
        meshes.append(riser)

    # Top platform at z=top_h, -X side, +Y strip
    top_box = trimesh.creation.box(
        (top_platform_len, flight_y_strip, top_h + 0.04),
        trimesh.transformations.translation_matrix(
            (top_platform_len / 2.0, flight2_y_center, (top_h + 0.04) / 2.0 - 0.02)
        ),
    )
    meshes.append(top_box)

    # Origin — spawn point on the apron (flight 1 start)
    origin = np.array([apron_len / 2.0, apron_y_center, 0.0])
    return meshes, origin


@configclass
class MeshOpenRiserSwitchbackCfg(SubTerrainBaseCfg):
    """Two-flight switchback, solid risers, no side walls between flights.

    Models the SM_Staircase_01 topology AFTER Colby's riser-baking — full
    architectural switchback with a landing platform between flights,
    solid riser per step on each flight, second flight 180° from first.
    The policy must learn the multi-stage ascent pattern: climb flight 1,
    walk across landing, turn around, climb flight 2.
    """

    function = open_riser_switchback_terrain

    step_height_range: tuple[float, float] = MISSING
    """[min, max] riser height (m) per flight. Phase-FW-Plus uses (0.15, 0.20)."""

    step_width: float = MISSING
    """Run depth per tread (m). Phase-FW-Plus uses 0.25 m to match FW."""

    approach_apron_len_m: float = 1.0
    """Flat apron at the spawn end (m)."""

    landing_len_m: float = 1.5
    """Length of the landing platform between flights (m). Spans full Y width."""

    top_platform_len_m: float = 1.0
    """Top platform at the end of flight 2 (m)."""

    flight_separation_m: float = 0.5
    """Gap between flight 1 and flight 2 in Y direction (m). Treated as void."""

    tread_thickness: float = 0.04
    """Vertical thickness of each tread slab (m). Auto-clamped to 40% of step_height."""

    riser_thickness: float = 0.02
    """Thickness of each solid riser face along X (m). 2 cm matches FW USD."""
