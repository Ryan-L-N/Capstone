"""
Diagnostic: Find Ground Plane Friction Prim
============================================
Run this ONCE to discover where IsaacSim stores the friction attributes
for the default ground plane in this version of IsaacSim.

It prints every prim in the stage, flags any with friction-related attributes,
and shows the exact path and attribute names to use in set_ground_friction().

Usage:
  ./isaacSim_env/Scripts/python.exe find_friction_prim.py --headless
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import omni
from omni.isaac.core import World
from pxr import UsdPhysics, Usd

world = World(physics_dt=1/500, rendering_dt=10/500, stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

world.scene.add_default_ground_plane(
    z_position=0,
    name="default_ground_plane",
    prim_path="/World/defaultGroundPlane",
    static_friction=0.75,
    dynamic_friction=0.65,
    restitution=0.01,
)

world.reset()

# Run a few steps so the ground plane is fully instantiated
for _ in range(5):
    world.step(render=False)

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FULL STAGE PRIM TREE")
print("=" * 70)

FRICTION_ATTR_NAMES = [
    "physics:staticFriction",
    "physics:dynamicFriction",
    "physxMaterial:staticFriction",
    "physxMaterial:dynamicFriction",
    "physxMaterial:restitution",
]

friction_prims_found = []

for prim in stage.Traverse():
    path = str(prim.GetPath())

    # Check for friction-related attributes
    attr_hits = []
    for attr_name in FRICTION_ATTR_NAMES:
        attr = prim.GetAttribute(attr_name)
        if attr.IsValid():
            attr_hits.append(f"{attr_name}={attr.Get()}")

    # Also check if UsdPhysics.MaterialAPI is applied
    has_material_api = False
    try:
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            has_material_api = True
            if not attr_hits:
                attr_hits.append("[UsdPhysics.MaterialAPI applied]")
    except Exception:
        pass

    if attr_hits:
        print(f"\n*** FRICTION PRIM FOUND ***")
        print(f"  Path:  {path}")
        print(f"  Type:  {prim.GetTypeName()}")
        for a in attr_hits:
            print(f"  Attr:  {a}")
        friction_prims_found.append(path)
    else:
        # Just print path for non-friction prims
        print(f"  {path}  [{prim.GetTypeName()}]")

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY — PRIMS WITH FRICTION ATTRIBUTES")
print("=" * 70)

if friction_prims_found:
    for p in friction_prims_found:
        print(f"  {p}")
    print(f"\nCopy one of these paths into set_ground_friction() in:")
    print(f"  baseline_runner.py")
    print(f"  training_env_1.py")
else:
    print("  None found via attribute names. Trying UsdPhysics.MaterialAPI scan...")
    for prim in stage.Traverse():
        try:
            mat = UsdPhysics.MaterialAPI(prim)
            sf = mat.GetStaticFrictionAttr()
            if sf and sf.IsValid():
                print(f"  Found via MaterialAPI: {prim.GetPath()}")
                friction_prims_found.append(str(prim.GetPath()))
        except Exception:
            pass

    if not friction_prims_found:
        print("  Still none found. The ground plane may use PhysX schema attributes.")
        print("  Look for prims containing 'Ground' or 'Physics' in path above.")

print("=" * 70)

simulation_app.close()
