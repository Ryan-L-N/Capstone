"""
Visual markers for target positions, goals, and waypoints.

Consolidates create_target_marker() from 10+ scripts.
Handles ES-004: checks for existing prims before creating.

Usage:
    from core import create_target_marker, create_goal_marker
    create_target_marker(stage, (17.3, 8.1, 0.0))
    create_goal_marker(stage, (5.0, 5.0, 0.0), color=(0, 0, 1))
"""


def create_target_marker(stage, position, name="target", color=(0.0, 1.0, 0.0),
                         radius=0.3, height=0.05):
    """
    Create a flat cylinder marker on the ground at the given position.

    Args:
        stage: USD stage
        position: (x, y) or (x, y, z) position
        name: Marker name (sanitized for USD path)
        color: RGB tuple (0-1)
        radius: Cylinder radius in meters
        height: Cylinder height in meters

    Returns:
        USD prim path of the marker
    """
    from pxr import UsdGeom, Gf

    safe_name = name.replace(".", "_").replace("-", "_").replace(" ", "_")
    marker_path = f"/World/{safe_name}_marker"

    # ES-004: Remove existing prim to prevent duplicates
    existing = stage.GetPrimAtPath(marker_path)
    if existing.IsValid():
        stage.RemovePrim(marker_path)

    marker = UsdGeom.Cylinder.Define(stage, marker_path)
    marker.GetRadiusAttr().Set(radius)
    marker.GetHeightAttr().Set(height)

    x = position[0]
    y = position[1]
    z = position[2] if len(position) > 2 else height / 2
    marker.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
    marker.GetDisplayColorAttr().Set([color])

    return marker_path


def create_goal_marker(stage, position, name="goal", color=(1.0, 0.0, 0.0),
                       size=0.5):
    """
    Create a sphere marker at the given position.

    Args:
        stage: USD stage
        position: (x, y) or (x, y, z) position
        name: Marker name
        color: RGB tuple (0-1)
        size: Sphere radius in meters

    Returns:
        USD prim path of the marker
    """
    from pxr import UsdGeom, Gf

    safe_name = name.replace(".", "_").replace("-", "_").replace(" ", "_")
    marker_path = f"/World/{safe_name}_marker"

    existing = stage.GetPrimAtPath(marker_path)
    if existing.IsValid():
        stage.RemovePrim(marker_path)

    marker = UsdGeom.Sphere.Define(stage, marker_path)
    marker.GetRadiusAttr().Set(size)

    x = position[0]
    y = position[1]
    z = position[2] if len(position) > 2 else size
    marker.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
    marker.GetDisplayColorAttr().Set([color])

    return marker_path
