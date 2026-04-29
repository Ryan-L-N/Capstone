#!/usr/bin/env python3
"""
MULTI-TERRAIN MULTI-ROBOT ENVIRONMENT - DEMONSTRATION
Showcasing 8 Spot robots across 8 different terrain types
No RL training or physics - visual demonstration only

Terrain Types:
1. Flat (baseline)
2. Beach Sand
3. Small Gravel (1/4" - 1/2")
4. Medium Gravel (1/2" - 1")
5. Large Gravel (1" - 2")
6. Mud
7. Thick Grass/Brush
8. Short Grass
"""

import argparse
import numpy as np
import random
from isaacsim import SimulationApp

# Create simulation app
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage
from omni.isaac.core.utils.prims import create_prim
from pxr import Gf, UsdGeom, UsdPhysics, Sdf
from omni.isaac.core.world import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import math

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Stage and world setup
stage = create_new_stage()
world = World(stage_units_in_meters=1.0)

print("=" * 70)
print("MULTI-TERRAIN MULTI-ROBOT ENVIRONMENT - DEMONSTRATION")
print("=" * 70)
print(f"Showcasing 8 Spot robots across 8 terrain types")
print(f"Environment: 110m long, 8 corridors × 15m wide = 120m total width")
print("=" * 70)

# =============================================================================
# TERRAIN CREATION FUNCTIONS
# =============================================================================

def create_flat_corridor(world, start_y, end_y, corridor_index):
    """Create flat baseline corridor"""
    print(f"Corridor {corridor_index}: FLAT TERRAIN")

def create_beach_sand(world, start_y, end_y, corridor_index):
    """Create beach sand obstacle"""
    print(f"Corridor {corridor_index}: BEACH SAND")
    sand = UsdGeom.Cube.Define(world.stage, f"/World/Sand_{corridor_index}")
    sand_xform = UsdGeom.Xformable(sand.GetPrim())
    sand_center_y = (start_y + end_y) / 2.0
    sand_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, sand_center_y, 0.08))
    sand_xform.AddScaleOp().Set(Gf.Vec3f(100.0, 15.0, 0.15))
    sand.GetDisplayColorAttr().Set([Gf.Vec3f(0.95, 0.88, 0.60)])

def create_small_gravel(world, start_y, end_y, corridor_index):
    """Create small gravel (1/4" - 1/2") obstacle"""
    print(f"Corridor {corridor_index}: SMALL GRAVEL (1/4\" - 1/2\")")
    create_gravel_corridor(world, start_y, end_y, corridor_index, 
                          pebble_min=0.00635, pebble_max=0.0127, color=(0.65, 0.58, 0.50))

def create_medium_gravel(world, start_y, end_y, corridor_index):
    """Create medium gravel (1/2" - 1") obstacle"""
    print(f"Corridor {corridor_index}: MEDIUM GRAVEL (1/2\" - 1\")")
    create_gravel_corridor(world, start_y, end_y, corridor_index,
                          pebble_min=0.0127, pebble_max=0.0254, color=(0.6, 0.5, 0.45))

def create_large_gravel(world, start_y, end_y, corridor_index):
    """Create large gravel (1" - 2") obstacle"""
    print(f"Corridor {corridor_index}: LARGE GRAVEL (1\" - 2\")")
    create_gravel_corridor(world, start_y, end_y, corridor_index,
                          pebble_min=0.0254, pebble_max=0.0508, color=(0.55, 0.48, 0.42))

def create_gravel_corridor(world, start_y, end_y, corridor_index, pebble_min, pebble_max, color):
    """Helper function to create gravel with varying pebble sizes"""
    # Base gravel area (visual only)
    gravel_base = UsdGeom.Cube.Define(world.stage, f"/World/GravelBase_{corridor_index}")
    gravel_xform = UsdGeom.Xformable(gravel_base.GetPrim())
    gravel_center_y = (start_y + end_y) / 2.0
    gravel_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, gravel_center_y, 0.08))
    gravel_xform.AddScaleOp().Set(Gf.Vec3f(100.0, 15.0, 0.15))
    gravel_base.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    
    # Scatter pebbles
    random.seed(42 + corridor_index)
    num_pebbles = 80
    for i in range(num_pebbles):
        pebble_x = random.uniform(-50.0, 50.0)
        pebble_y = random.uniform(start_y + 1.0, end_y - 1.0)
        pebble_size = random.uniform(pebble_min, pebble_max)
        pebble_z = 0.1 + pebble_size
        
        pebble = UsdGeom.Sphere.Define(world.stage, f"/World/Pebble_{corridor_index}_{i}")
        pebble_xform = UsdGeom.Xformable(pebble.GetPrim())
        pebble_xform.AddTranslateOp().Set(Gf.Vec3d(pebble_x, pebble_y, pebble_z))
        pebble_xform.AddScaleOp().Set(Gf.Vec3f(pebble_size, pebble_size, pebble_size))
        pebble.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

def create_mud(world, start_y, end_y, corridor_index):
    """Create mud obstacle (high friction)"""
    print(f"Corridor {corridor_index}: MUD")
    mud = UsdGeom.Cube.Define(world.stage, f"/World/Mud_{corridor_index}")
    mud_xform = UsdGeom.Xformable(mud.GetPrim())
    mud_center_y = (start_y + end_y) / 2.0
    mud_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, mud_center_y, 0.08))
    mud_xform.AddScaleOp().Set(Gf.Vec3f(100.0, 15.0, 0.15))
    mud.GetDisplayColorAttr().Set([Gf.Vec3f(0.4, 0.3, 0.2)])

def create_thick_grass_brush(world, start_y, end_y, corridor_index):
    """Create tall grass and brush"""
    print(f"Corridor {corridor_index}: THICK GRASS & BRUSH")
    random.seed(42 + corridor_index)
    num_obstacles = 100
    for i in range(num_obstacles):
        obs_x = random.uniform(-50.0, 50.0)
        obs_y = random.uniform(start_y + 1.0, end_y - 1.0)
        obs_height = random.uniform(0.5, 1.2)
        obs_width = random.uniform(0.1, 0.3)
        
        grass = UsdGeom.Cylinder.Define(world.stage, f"/World/Grass_{corridor_index}_{i}")
        grass_xform = UsdGeom.Xformable(grass.GetPrim())
        grass_xform.AddTranslateOp().Set(Gf.Vec3d(obs_x, obs_y, obs_height / 2.0))
        grass_xform.AddScaleOp().Set(Gf.Vec3f(obs_width, obs_width, obs_height))
        grass.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.5, 0.2)])

def create_short_grass(world, start_y, end_y, corridor_index):
    """Create short grass (low resistance, tall appearance)"""
    print(f"Corridor {corridor_index}: SHORT GRASS")
    random.seed(42 + corridor_index)
    num_obstacles = 150
    for i in range(num_obstacles):
        obs_x = random.uniform(-50.0, 50.0)
        obs_y = random.uniform(start_y + 1.0, end_y - 1.0)
        obs_height = random.uniform(0.15, 0.35)
        obs_width = random.uniform(0.05, 0.15)
        
        grass = UsdGeom.Cylinder.Define(world.stage, f"/World/ShortGrass_{corridor_index}_{i}")
        grass_xform = UsdGeom.Xformable(grass.GetPrim())
        grass_xform.AddTranslateOp().Set(Gf.Vec3d(obs_x, obs_y, obs_height / 2.0))
        grass_xform.AddScaleOp().Set(Gf.Vec3f(obs_width, obs_width, obs_height))
        grass.GetDisplayColorAttr().Set([Gf.Vec3f(0.3, 0.6, 0.2)])

# =============================================================================
# CREATE WORLD SETUP
# =============================================================================

# Ground plane
ground_plane = UsdGeom.Cube.Define(world.stage, "/World/GroundPlane")
ground_xform = UsdGeom.Xformable(ground_plane.GetPrim())
ground_xform.AddTranslateOp().Set(Gf.Vec3d(0, 60, -0.1))
ground_xform.AddScaleOp().Set(Gf.Vec3f(110.0, 120.0, 0.2))
ground_plane.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 0.0)])

# Create 8 parallel corridors with terrain
corridor_configs = [
    ("Flat", create_flat_corridor),
    ("Beach Sand", create_beach_sand),
    ("Small Gravel", create_small_gravel),
    ("Medium Gravel", create_medium_gravel),
    ("Large Gravel", create_large_gravel),
    ("Mud", create_mud),
    ("Thick Grass", create_thick_grass_brush),
    ("Short Grass", create_short_grass),
]

corridor_y_positions = []
for i in range(8):
    start_y = i * 15.0
    end_y = (i + 1) * 15.0
    corridor_y_positions.append((start_y, end_y))
    
    # Create corridor markers (clear lines of separation)
    marker_left = UsdGeom.Cube.Define(world.stage, f"/World/CorridorMarker_{i}_left")
    marker_xform = UsdGeom.Xformable(marker_left.GetPrim())
    marker_xform.AddTranslateOp().Set(Gf.Vec3d(-55.0, (start_y + end_y) / 2.0 - 0.25, 0.0))
    marker_xform.AddScaleOp().Set(Gf.Vec3f(110.0, 0.5, 0.02))
    marker_left.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 1.0, 1.0)])
    
    # Create terrain for this corridor
    terrain_func = corridor_configs[i][1]
    terrain_func(world, start_y, end_y, i)
    
    # Create start zone (green 1m × 1m square)
    start_zone = UsdGeom.Cube.Define(world.stage, f"/World/StartZone_{i}")
    start_xform = UsdGeom.Xformable(start_zone.GetPrim())
    start_xform.AddTranslateOp().Set(Gf.Vec3d(-52.5, (start_y + end_y) / 2.0, 0.05))
    start_xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 0.1))
    start_zone.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.8, 0.2)])
    
    # Create end zone (black sphere with red flag)
    end_sphere = UsdGeom.Sphere.Define(world.stage, f"/World/EndZone_{i}")
    end_xform = UsdGeom.Xformable(end_sphere.GetPrim())
    end_xform.AddTranslateOp().Set(Gf.Vec3d(52.5, (start_y + end_y) / 2.0, 1.0))
    end_xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    end_sphere.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 0.0)])
    
    # Red flag
    flag = UsdGeom.Cylinder.Define(world.stage, f"/World/Flag_{i}")
    flag_xform = UsdGeom.Xformable(flag.GetPrim())
    flag_xform.AddTranslateOp().Set(Gf.Vec3d(52.5, (start_y + end_y) / 2.0, 2.0))
    flag_xform.AddScaleOp().Set(Gf.Vec3f(0.1, 0.5, 0.1))
    flag.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])

print("Environment created with 8 parallel corridors")
print("=" * 70)

# =============================================================================
# SPAWN 8 SPOT ROBOTS
# =============================================================================

spot_robots = []
for i in range(8):
    robot_name = f"Spot_{i}"
    start_y = corridor_y_positions[i][0]
    end_y = corridor_y_positions[i][1]
    spawn_y = (start_y + end_y) / 2.0
    
    # Add Spot robot
    spot_path = f"/World/{robot_name}"
    add_reference_to_stage(
        usd_path="omniverse://localhost/Projects/Robotics/Robots/Spot/spot.usd",
        prim_path=spot_path
    )
    
    robot = Robot(prim_path=spot_path, name=robot_name)
    robot.set_world_pose(position=[-52.5, spawn_y, 0.5], orientation=[0, 0, 0, 1])
    spot_robots.append(robot)

print(f"Spawned 8 Spot robots")
print("=" * 70)
print("\nEnvironment ready for demonstration!")
print("8 corridors with different terrain types")
print("8 Spot robots ready for deployment")
print("=" * 70)

# =============================================================================
# MAIN VISUALIZATION LOOP
# =============================================================================

def run_demo(duration_seconds=60):
    """Run demonstration with basic visualization"""
    import time
    
    # Initialize physics for rendering
    world.initialize_physics()
    
    start_time = time.time()
    step = 0
    
    print(f"\nRunning {duration_seconds}s demonstration...")
    print("Use mouse to rotate/pan camera in GUI")
    print("Press ESC to exit\n")
    
    while (time.time() - start_time) < duration_seconds:
        world.step(render=True)
        step += 1
        
        # Print progress every 100 steps
        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  {elapsed:.1f}s elapsed... ({step} simulation steps)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    args = parser.parse_args()
    
    try:
        run_demo(duration_seconds=args.duration)
    finally:
        simulation_app.close()
