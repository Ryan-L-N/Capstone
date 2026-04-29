"""
Lighting presets for Isaac Sim scenes.

Consolidates create_room_lighting() duplicated across 10+ scripts.

Usage:
    from core import create_room_lighting, LightingPreset
    create_room_lighting(stage, room_length=18.3, room_width=9.1)
    create_room_lighting(stage, preset=LightingPreset.MINIMAL)
"""

from enum import Enum


class LightingPreset(Enum):
    INDOOR_ROOM = "indoor_room"
    OUTDOOR = "outdoor"
    MINIMAL = "minimal"


def create_room_lighting(stage, room_length=18.3, room_width=9.1,
                         preset=LightingPreset.INDOOR_ROOM):
    """
    Create scene lighting with configurable presets.

    Args:
        stage: USD stage
        room_length: Room X dimension (meters)
        room_width: Room Y dimension (meters)
        preset: LightingPreset enum value

    Returns:
        Root prim path for lights
    """
    from pxr import UsdGeom, UsdLux, Gf

    lights_path = "/World/Lights"

    if stage.GetPrimAtPath(lights_path).IsValid():
        return lights_path

    UsdGeom.Xform.Define(stage, lights_path)

    # Dome light (always present)
    dome = UsdLux.DomeLight.Define(stage, f"{lights_path}/DomeLight")
    dome.CreateIntensityAttr(500.0)
    dome.CreateTextureFormatAttr("latlong")

    if preset == LightingPreset.MINIMAL:
        print("Lighting created: Dome only (minimal)")
        return lights_path

    # Distant light (sun) for directional shadows
    sun = UsdLux.DistantLight.Define(stage, f"{lights_path}/SunLight")
    sun.CreateAngleAttr(1.0)

    if preset == LightingPreset.OUTDOOR:
        sun.CreateIntensityAttr(8000.0)
        sun_prim = stage.GetPrimAtPath(f"{lights_path}/SunLight")
        UsdGeom.Xformable(sun_prim).AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))
        print("Lighting created: Dome + Sun (outdoor)")
        return lights_path

    # Indoor room: dome + sun + ceiling rect light
    sun.CreateIntensityAttr(5000.0)
    sun_prim = stage.GetPrimAtPath(f"{lights_path}/SunLight")
    UsdGeom.Xformable(sun_prim).AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    rect = UsdLux.RectLight.Define(stage, f"{lights_path}/CeilingLight")
    rect.CreateIntensityAttr(3000.0)
    rect.CreateWidthAttr(room_length * 0.8)
    rect.CreateHeightAttr(room_width * 0.8)
    rect_prim = stage.GetPrimAtPath(f"{lights_path}/CeilingLight")
    xf = UsdGeom.Xformable(rect_prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(room_length / 2, room_width / 2, 2.8))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(180, 0, 0))

    print("Lighting created: Dome + Sun + Ceiling (indoor)")
    return lights_path
