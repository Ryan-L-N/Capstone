"""
Grass Physics Configuration Module - ES-010: Proper PhysX Grass Implementation
===============================================================================

This module implements RL-friendly grass physics using PhysX materials and
sparse proxy stalks instead of dense collision bodies.

PROBLEM SOLVED:
- Previous implementation: 1166+ collision cylinders → 100% robot fall rate
- New implementation: Friction-based ground + sparse proxy stalks → stable physics

KEY CONCEPTS:
1. GrassMaterial: PhysX material with tuned friction/damping for grass terrain
2. Proxy Stalks: Sparse (~0.5-1.0/ft²) kinematic colliders for brush feedback
3. ENABLE_PROXY_STALKS toggle: Switch between friction-only and stalks+friction

PHYSICS EXPLANATION - Friction Combine Modes:
- PhysX uses "average" mode by default for friction
- Robot foot: μ_static ≈ 0.55, μ_dynamic ≈ 0.45
- GrassMaterial: μ_static = 0.20, μ_dynamic = 0.15
- Effective friction = (0.55 + 0.20) / 2 = 0.375 static
- Effective friction = (0.45 + 0.15) / 2 = 0.30 dynamic
- This gives realistic grass slip behavior (~0.30-0.40 range)

Room: 30ft × 60ft = 9.14m × 18.29m = 167.2 m² = 1800 ft²

Isaac Sim 5.1.0 Compatible
"""

import numpy as np
import random
from pxr import UsdGeom, Gf, UsdPhysics, UsdShade, Sdf, PhysxSchema


# =============================================================================
# CONFIGURATION FLAGS
# =============================================================================

# Master toggle for proxy stalk collision
ENABLE_PROXY_STALKS = True  # Set False for friction-only mode (better performance)

# Room dimensions (60ft x 30ft = 18.29m x 9.14m)
ROOM_LENGTH_FT = 60
ROOM_WIDTH_FT = 30
ROOM_LENGTH_M = 18.29  # meters
ROOM_WIDTH_M = 9.14    # meters
ROOM_AREA_FT2 = ROOM_LENGTH_FT * ROOM_WIDTH_FT  # 1800 ft²
ROOM_AREA_M2 = ROOM_LENGTH_M * ROOM_WIDTH_M      # 167.2 m²

# =============================================================================
# GRASS MATERIAL CONFIGURATION (Section A)
# =============================================================================

GRASS_MATERIAL_CONFIG = {
    "name": "GrassMaterial",
    "path": "/World/Physics/Materials/GrassMaterial",
    
    # Friction values - designed to combine with robot foot friction
    # Robot foot: static ~0.55, dynamic ~0.45
    # Baseline floor uses 0.8 friction - grass should provide similar or higher
    # Combined (average): static ~0.675, dynamic ~0.575
    # These values allow stable walking while simulating grass texture
    "static_friction": 0.80,   # Match baseline - grass provides good grip
    "dynamic_friction": 0.70,  # Slightly less than static for realistic slip
    "restitution": 0.05,  # Very low bounce
    
    # PhysX combine modes
    "friction_combine_mode": "average",      # (grass + foot) / 2
    "restitution_combine_mode": "min",       # min(grass, foot) = minimal bounce
}

# Robot foot material (for reference - applied to robot, not grass)
ROBOT_FOOT_MATERIAL = {
    "static_friction": 0.55,
    "dynamic_friction": 0.45,
    "restitution": 0.1,
}

# =============================================================================
# PROXY STALK CONFIGURATION (Section B)
# =============================================================================

# Density options (stalks per square foot)
STALK_DENSITY_OPTIONS = {
    "sparse": 0.5,   # 900 stalks for 1800 ft² - RECOMMENDED for performance
    "medium": 1.0,   # 1800 stalks for 1800 ft²
    "dense": 2.0,    # 3600 stalks - NOT recommended, use friction instead
}

# Selected density (0.5/ft² recommended for stability + performance)
STALK_DENSITY_PER_FT2 = 0.5  # Results in ~900 stalks for 1800 ft²

PROXY_STALK_CONFIG = {
    # Geometry
    "height_min": 0.30,      # meters
    "height_max": 0.50,      # meters
    "radius_min": 0.005,     # meters (5mm)
    "radius_max": 0.015,     # meters (15mm)
    "base_sink": 0.02,       # Sink base 2cm into ground to prevent jitter
    
    # Physics
    "use_kinematic": True,   # Kinematic = stable, no physics solve needed
    "collision_enabled": True,
    "self_collision": False, # CRITICAL: Disable stalk-stalk collision
    
    # If using dynamic stalks (not recommended)
    "dynamic_mass": 0.001,   # Very low mass (1 gram)
    "linear_damping": 10.0,  # Heavy damping
    "angular_damping": 10.0,
    
    # Material
    "use_grass_material": True,  # Apply GrassMaterial to stalks
}

# =============================================================================
# GRASS ZONE CONFIGURATION
# =============================================================================

# Default grass zone (can be overridden)
DEFAULT_GRASS_ZONE = {
    "x_min": 2.0,    # meters from origin
    "x_max": 16.3,   # meters
    "y_min": 2.0,    # meters
    "y_max": 7.1,    # meters
}


# =============================================================================
# PHYSX MATERIAL CREATION FUNCTIONS
# =============================================================================

def create_grass_material(stage, material_path=None):
    """
    Create the GrassMaterial PhysX material with proper friction settings.
    
    This is the PRIMARY grass effect - friction-based slowdown on ground contact.
    
    Args:
        stage: USD stage
        material_path: Optional custom path, defaults to config path
        
    Returns:
        tuple: (material_path, material_prim)
    """
    if material_path is None:
        material_path = GRASS_MATERIAL_CONFIG["path"]
    
    # Ensure parent path exists
    parent_path = "/".join(material_path.rsplit("/", 1)[:-1]) or "/World/Physics/Materials"
    if not stage.GetPrimAtPath(parent_path).IsValid():
        UsdGeom.Xform.Define(stage, parent_path)
    
    # Check for existing material
    existing = stage.GetPrimAtPath(material_path)
    if existing.IsValid():
        print(f"  [GrassMaterial] Updating existing material at {material_path}")
        material_prim = existing
    else:
        print(f"  [GrassMaterial] Creating new material at {material_path}")
        material = UsdShade.Material.Define(stage, material_path)
        material_prim = material.GetPrim()
    
    # Apply PhysX material API
    physics_material = UsdPhysics.MaterialAPI.Apply(material_prim)
    
    # Set friction values
    physics_material.CreateStaticFrictionAttr(GRASS_MATERIAL_CONFIG["static_friction"])
    physics_material.CreateDynamicFrictionAttr(GRASS_MATERIAL_CONFIG["dynamic_friction"])
    physics_material.CreateRestitutionAttr(GRASS_MATERIAL_CONFIG["restitution"])
    
    # Apply PhysxSchema for combine modes
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
    
    # Set combine modes using TfToken strings
    # PhysX expects these as token strings: "average", "min", "multiply", "max"
    from pxr import Sdf
    friction_mode = GRASS_MATERIAL_CONFIG["friction_combine_mode"]
    restitution_mode = GRASS_MATERIAL_CONFIG["restitution_combine_mode"]
    
    physx_material.CreateFrictionCombineModeAttr().Set(friction_mode)
    physx_material.CreateRestitutionCombineModeAttr().Set(restitution_mode)
    
    print(f"  [GrassMaterial] Properties set:")
    print(f"    Static friction:  {GRASS_MATERIAL_CONFIG['static_friction']}")
    print(f"    Dynamic friction: {GRASS_MATERIAL_CONFIG['dynamic_friction']}")
    print(f"    Restitution:      {GRASS_MATERIAL_CONFIG['restitution']}")
    print(f"    Friction combine: {GRASS_MATERIAL_CONFIG['friction_combine_mode']}")
    print(f"    Effective friction (mu) with robot foot (~0.55/0.45):")
    eff_static = (GRASS_MATERIAL_CONFIG['static_friction'] + ROBOT_FOOT_MATERIAL['static_friction']) / 2
    eff_dynamic = (GRASS_MATERIAL_CONFIG['dynamic_friction'] + ROBOT_FOOT_MATERIAL['dynamic_friction']) / 2
    print(f"      Static:  ({GRASS_MATERIAL_CONFIG['static_friction']} + 0.55) / 2 = {eff_static:.3f}")
    print(f"      Dynamic: ({GRASS_MATERIAL_CONFIG['dynamic_friction']} + 0.45) / 2 = {eff_dynamic:.3f}")
    
    return material_path, material_prim


def apply_material_to_prim(stage, prim_path, material_path):
    """
    Apply a PhysX material to a collision prim.
    
    Args:
        stage: USD stage
        prim_path: Path to the prim to modify
        material_path: Path to the material to apply
        
    Returns:
        bool: Success status
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"  [Warning] Prim not found: {prim_path}")
        return False
    
    material = UsdShade.Material.Get(stage, material_path)
    if not material:
        print(f"  [Warning] Material not found: {material_path}")
        return False
    
    # Apply material binding
    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    binding_api.Bind(material)
    
    return True


def apply_grass_material_to_ground(stage, ground_prim_path="/World/Ground"):
    """
    Apply GrassMaterial to the ground collision prim.
    
    This is the MAIN grass physics effect - the robot experiences grass-like
    friction when walking on the ground.
    
    Args:
        stage: USD stage
        ground_prim_path: Path to ground collision prim
        
    Returns:
        bool: Success status
    """
    print(f"\n[Applying GrassMaterial to Ground]")
    
    # Create grass material if it doesn't exist
    material_path, _ = create_grass_material(stage)
    
    # Find ground prim
    ground_prim = stage.GetPrimAtPath(ground_prim_path)
    if not ground_prim.IsValid():
        # Try common alternative paths
        alternatives = [
            "/World/GroundPlane",
            "/World/defaultGroundPlane",
            "/World/Ground/Collision",
            "/World/groundPlane",
        ]
        for alt in alternatives:
            ground_prim = stage.GetPrimAtPath(alt)
            if ground_prim.IsValid():
                ground_prim_path = alt
                break
    
    if not ground_prim.IsValid():
        print(f"  [Error] Ground prim not found at {ground_prim_path}")
        print(f"  Searched alternatives: {alternatives}")
        return False
    
    print(f"  Found ground prim: {ground_prim_path}")
    
    # Ensure collision is enabled on ground
    if not UsdPhysics.CollisionAPI.Get(stage, ground_prim_path):
        UsdPhysics.CollisionAPI.Apply(ground_prim)
        print(f"  Applied CollisionAPI to ground")
    
    # Apply material
    success = apply_material_to_prim(stage, ground_prim_path, material_path)
    if success:
        print(f"  ✓ GrassMaterial applied to ground: {ground_prim_path}")
    
    return success


# =============================================================================
# PROXY STALK MANAGEMENT FUNCTIONS
# =============================================================================

def count_existing_stalks(stage, stalk_patterns=None):
    """
    Count existing stalk prims in the scene.
    
    Args:
        stage: USD stage
        stalk_patterns: List of path patterns to search (default: common patterns)
        
    Returns:
        tuple: (count, list of stalk paths)
    """
    if stalk_patterns is None:
        stalk_patterns = [
            "/World/GrassZone/cluster_",
            "/World/Grass/stalk_",
            "/World/grass_",
            "/World/GrassZone/stalk_",
        ]
    
    stalk_paths = []
    
    # Search for stalks under common parent paths
    search_paths = ["/World/GrassZone", "/World/Grass", "/World"]
    
    for search_path in search_paths:
        parent = stage.GetPrimAtPath(search_path)
        if not parent.IsValid():
            continue
            
        for child in parent.GetAllChildren():
            child_path = str(child.GetPath())
            # Check if it looks like a stalk
            if any(pattern in child_path.lower() for pattern in ["cluster", "stalk", "grass_", "blade"]):
                if child.IsA(UsdGeom.Cylinder) or child.IsA(UsdGeom.Capsule):
                    stalk_paths.append(child_path)
    
    return len(stalk_paths), stalk_paths


def thin_stalks_to_target_density(stage, target_density=None, grass_zone=None, seed=42):
    """
    Reduce existing stalks to target density by disabling collision on extras.
    
    For 1800 ft² room:
    - 0.5 stalks/ft² = 900 stalks with collision
    - 1.0 stalks/ft² = 1800 stalks with collision
    
    Args:
        stage: USD stage
        target_density: Stalks per ft² (default from config)
        grass_zone: Zone bounds dict (default from config)
        seed: Random seed for consistent thinning
        
    Returns:
        tuple: (enabled_count, disabled_count, kept_paths)
    """
    if target_density is None:
        target_density = STALK_DENSITY_PER_FT2
    if grass_zone is None:
        grass_zone = DEFAULT_GRASS_ZONE
    
    print(f"\n[Thinning Stalks to Target Density]")
    print(f"  Target density: {target_density} stalks/ft²")
    
    # Calculate zone area in ft²
    zone_width_m = grass_zone["x_max"] - grass_zone["x_min"]
    zone_depth_m = grass_zone["y_max"] - grass_zone["y_min"]
    zone_area_m2 = zone_width_m * zone_depth_m
    zone_area_ft2 = zone_area_m2 * 10.764  # m² to ft²
    
    target_count = int(zone_area_ft2 * target_density)
    print(f"  Zone area: {zone_area_ft2:.1f} ft² ({zone_area_m2:.1f} m²)")
    print(f"  Target stalk count: {target_count}")
    
    # Find existing stalks
    current_count, stalk_paths = count_existing_stalks(stage)
    print(f"  Current stalk count: {current_count}")
    
    if current_count == 0:
        print(f"  No existing stalks found")
        return 0, 0, []
    
    # Set random seed for reproducible thinning
    random.seed(seed)
    
    # Shuffle and select stalks to keep
    shuffled_paths = stalk_paths.copy()
    random.shuffle(shuffled_paths)
    
    stalks_to_keep = shuffled_paths[:target_count]
    stalks_to_disable = shuffled_paths[target_count:]
    
    enabled_count = 0
    disabled_count = 0
    
    # Enable collision on kept stalks, disable on extras
    for path in stalks_to_keep:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            if ENABLE_PROXY_STALKS:
                # Ensure collision is enabled
                if not UsdPhysics.CollisionAPI.Get(stage, path):
                    UsdPhysics.CollisionAPI.Apply(prim)
                enabled_count += 1
            else:
                # Disable collision (friction-only mode)
                collision_api = UsdPhysics.CollisionAPI.Get(stage, path)
                if collision_api:
                    collision_api.GetCollisionEnabledAttr().Set(False)
                disabled_count += 1
    
    # Disable collision on extra stalks
    for path in stalks_to_disable:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            collision_api = UsdPhysics.CollisionAPI.Get(stage, path)
            if collision_api:
                collision_api.GetCollisionEnabledAttr().Set(False)
            disabled_count += 1
    
    print(f"  Enabled collision: {enabled_count} stalks")
    print(f"  Disabled collision: {disabled_count} stalks")
    
    return enabled_count, disabled_count, stalks_to_keep


def configure_proxy_stalk_physics(stage, stalk_paths, material_path=None):
    """
    Configure physics settings for proxy stalks.
    
    Settings applied:
    - Kinematic rigid body (stable, no physics solve)
    - GrassMaterial for friction
    - Self-collision disabled
    - Base sunk into ground
    
    Args:
        stage: USD stage
        stalk_paths: List of stalk prim paths
        material_path: Path to physics material (default: GrassMaterial)
        
    Returns:
        int: Number of stalks configured
    """
    if material_path is None:
        material_path = GRASS_MATERIAL_CONFIG["path"]
    
    print(f"\n[Configuring Proxy Stalk Physics]")
    print(f"  Mode: {'Kinematic' if PROXY_STALK_CONFIG['use_kinematic'] else 'Dynamic'}")
    print(f"  Self-collision: {PROXY_STALK_CONFIG['self_collision']}")
    
    configured = 0
    
    for path in stalk_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        
        # Get or create geom
        geom = UsdGeom.Gprim.Get(stage, path)
        if not geom:
            continue
        
        # Adjust height to sink base into ground
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                current_pos = op.Get()
                # Sink by base_sink amount
                new_pos = Gf.Vec3d(current_pos[0], current_pos[1], 
                                   current_pos[2] - PROXY_STALK_CONFIG["base_sink"])
                op.Set(new_pos)
                break
        
        # Apply rigid body API
        rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
        
        if PROXY_STALK_CONFIG["use_kinematic"]:
            # Kinematic = stable, no physics solve
            rigid_body.CreateKinematicEnabledAttr(True)
        else:
            # Dynamic with heavy damping (not recommended)
            rigid_body.CreateKinematicEnabledAttr(False)
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(PROXY_STALK_CONFIG["dynamic_mass"])
            
            # Apply damping via PhysxRigidBodyAPI
            physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            physx_rb.CreateLinearDampingAttr(PROXY_STALK_CONFIG["linear_damping"])
            physx_rb.CreateAngularDampingAttr(PROXY_STALK_CONFIG["angular_damping"])
        
        # Apply collision with self-collision disabled
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        
        # Apply material
        if PROXY_STALK_CONFIG["use_grass_material"]:
            apply_material_to_prim(stage, path, material_path)
        
        configured += 1
    
    print(f"  Configured {configured} stalks")
    
    return configured


def disable_stalk_self_collision(stage, stalk_paths):
    """
    Disable collision between stalks using a collision filter group.
    
    This prevents stalks from colliding with each other while still
    allowing them to collide with the robot.
    
    Args:
        stage: USD stage
        stalk_paths: List of stalk prim paths
        
    Returns:
        bool: Success status
    """
    print(f"\n[Disabling Stalk Self-Collision]")
    
    # Create collision group for stalks
    group_path = "/World/Physics/StalkCollisionGroup"
    
    # Check if collision group exists
    existing = stage.GetPrimAtPath(group_path)
    if existing.IsValid():
        stage.RemovePrim(group_path)
    
    # Create collision group
    UsdGeom.Xform.Define(stage, group_path)
    group_prim = stage.GetPrimAtPath(group_path)
    
    # Apply collision group API
    collision_group = UsdPhysics.CollisionGroup.Define(stage, group_path)
    
    # Create filtered relationship to self (stalks don't collide with stalks)
    filtered_rel = collision_group.CreateFilteredGroupsRel()
    filtered_rel.AddTarget(group_path)  # Filter collision with itself
    
    # Add all stalks to this group
    collection = collision_group.GetCollidersCollectionAPI()
    includes_rel = collection.CreateIncludesRel()
    
    for path in stalk_paths:
        includes_rel.AddTarget(path)
    
    print(f"  Created collision group: {group_path}")
    print(f"  Added {len(stalk_paths)} stalks to filtered group")
    print(f"  Stalks will collide with robot but NOT with each other")
    
    return True


# =============================================================================
# MAIN SETUP FUNCTION
# =============================================================================

def setup_grass_physics(stage, grass_zone=None, ground_path="/World/Ground", seed=42):
    """
    Complete grass physics setup for the room.
    
    This function:
    1. Creates/applies GrassMaterial to ground
    2. Finds and thins existing stalks to target density
    3. Configures proxy stalk physics
    4. Disables stalk self-collision
    
    Args:
        stage: USD stage
        grass_zone: Zone bounds dict (default from config)
        ground_path: Path to ground prim
        seed: Random seed for reproducible results
        
    Returns:
        dict: Setup results with counts and paths
    """
    print("=" * 70)
    print("GRASS PHYSICS SETUP - ES-010")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  ENABLE_PROXY_STALKS: {ENABLE_PROXY_STALKS}")
    print(f"  Target density: {STALK_DENSITY_PER_FT2} stalks/ft²")
    print(f"  Room area: {ROOM_AREA_FT2} ft²")
    print(f"  Expected stalks: {int(ROOM_AREA_FT2 * STALK_DENSITY_PER_FT2)}")
    
    results = {
        "material_created": False,
        "material_path": None,
        "ground_material_applied": False,
        "stalks_enabled": 0,
        "stalks_disabled": 0,
        "stalks_configured": 0,
        "self_collision_disabled": False,
    }
    
    # Step 1: Create GrassMaterial
    print("\n" + "-" * 40)
    print("Step 1: Creating GrassMaterial")
    print("-" * 40)
    material_path, _ = create_grass_material(stage)
    results["material_created"] = True
    results["material_path"] = material_path
    
    # Step 2: Apply to ground
    print("\n" + "-" * 40)
    print("Step 2: Applying GrassMaterial to Ground")
    print("-" * 40)
    results["ground_material_applied"] = apply_grass_material_to_ground(stage, ground_path)
    
    # Step 3: Thin stalks to target density
    print("\n" + "-" * 40)
    print("Step 3: Thinning Stalks to Target Density")
    print("-" * 40)
    enabled, disabled, kept_paths = thin_stalks_to_target_density(
        stage, 
        target_density=STALK_DENSITY_PER_FT2,
        grass_zone=grass_zone,
        seed=seed
    )
    results["stalks_enabled"] = enabled
    results["stalks_disabled"] = disabled
    
    # Step 4: Configure proxy stalk physics (only if enabled)
    if ENABLE_PROXY_STALKS and kept_paths:
        print("\n" + "-" * 40)
        print("Step 4: Configuring Proxy Stalk Physics")
        print("-" * 40)
        results["stalks_configured"] = configure_proxy_stalk_physics(
            stage, kept_paths, material_path
        )
        
        # Step 5: Disable self-collision
        print("\n" + "-" * 40)
        print("Step 5: Disabling Stalk Self-Collision")
        print("-" * 40)
        results["self_collision_disabled"] = disable_stalk_self_collision(
            stage, kept_paths
        )
    else:
        print("\n[Skipping proxy stalk physics - ENABLE_PROXY_STALKS is False]")
    
    # Summary
    print("\n" + "=" * 70)
    print("GRASS PHYSICS SETUP COMPLETE")
    print("=" * 70)
    print(f"\nResults:")
    print(f"  ✓ GrassMaterial created: {results['material_path']}")
    print(f"  ✓ Ground material applied: {results['ground_material_applied']}")
    print(f"  ✓ Stalks with collision: {results['stalks_enabled']}")
    print(f"  ✓ Stalks collision disabled: {results['stalks_disabled']}")
    print(f"  ✓ Stalks configured: {results['stalks_configured']}")
    print(f"  ✓ Self-collision disabled: {results['self_collision_disabled']}")
    
    return results


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_grass_physics(stage, ground_path="/World/Ground"):
    """
    Validate grass physics setup.
    
    Checks:
    1. GrassMaterial exists with correct properties
    2. Ground has material applied
    3. Stalk count is within target range
    4. Self-collision is disabled
    
    Returns:
        dict: Validation results
    """
    print("\n" + "=" * 70)
    print("GRASS PHYSICS VALIDATION")
    print("=" * 70)
    
    results = {
        "material_valid": False,
        "ground_valid": False,
        "stalk_count_valid": False,
        "self_collision_valid": False,
        "issues": [],
    }
    
    # Check material
    material_path = GRASS_MATERIAL_CONFIG["path"]
    material_prim = stage.GetPrimAtPath(material_path)
    if material_prim.IsValid():
        physics_mat = UsdPhysics.MaterialAPI.Get(stage, material_path)
        if physics_mat:
            static_f = physics_mat.GetStaticFrictionAttr().Get()
            dynamic_f = physics_mat.GetDynamicFrictionAttr().Get()
            
            if abs(static_f - GRASS_MATERIAL_CONFIG["static_friction"]) < 0.01:
                results["material_valid"] = True
                print(f"  [OK] GrassMaterial valid: static={static_f}, dynamic={dynamic_f}")
            else:
                results["issues"].append(f"Wrong friction values: {static_f}, {dynamic_f}")
    else:
        results["issues"].append(f"GrassMaterial not found at {material_path}")
    
    # Check ground
    ground_prim = stage.GetPrimAtPath(ground_path)
    if ground_prim.IsValid():
        binding = UsdShade.MaterialBindingAPI.Get(stage, ground_path)
        if binding:
            bound_mat = binding.GetDirectBinding().GetMaterialPath()
            if bound_mat == material_path:
                results["ground_valid"] = True
                print(f"  ✓ Ground has GrassMaterial applied")
            else:
                results["issues"].append(f"Ground has wrong material: {bound_mat}")
    else:
        results["issues"].append(f"Ground prim not found: {ground_path}")
    
    # Check stalk count
    stalk_count, stalk_paths = count_existing_stalks(stage)
    enabled_count = 0
    for path in stalk_paths:
        collision = UsdPhysics.CollisionAPI.Get(stage, path)
        if collision and collision.GetCollisionEnabledAttr().Get() != False:
            enabled_count += 1
    
    target = int(ROOM_AREA_FT2 * STALK_DENSITY_PER_FT2)
    tolerance = 0.2 * target  # 20% tolerance
    
    if abs(enabled_count - target) < tolerance or not ENABLE_PROXY_STALKS:
        results["stalk_count_valid"] = True
        print(f"  ✓ Stalk count valid: {enabled_count}/{stalk_count} (target: {target})")
    else:
        results["issues"].append(f"Stalk count mismatch: {enabled_count} vs target {target}")
    
    # Check self-collision
    group_path = "/World/Physics/StalkCollisionGroup"
    group_prim = stage.GetPrimAtPath(group_path)
    if group_prim.IsValid() or not ENABLE_PROXY_STALKS:
        results["self_collision_valid"] = True
        print(f"  ✓ Self-collision configuration valid")
    else:
        results["issues"].append("Stalk collision group not found")
    
    # Summary
    all_valid = all([
        results["material_valid"],
        results["ground_valid"],
        results["stalk_count_valid"],
        results["self_collision_valid"],
    ])
    
    print(f"\nOverall: {'✓ VALID' if all_valid else '✗ ISSUES FOUND'}")
    if results["issues"]:
        print("Issues:")
        for issue in results["issues"]:
            print(f"  - {issue}")
    
    return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def set_proxy_stalks_enabled(enabled: bool):
    """
    Toggle proxy stalk collision on/off.
    
    Args:
        enabled: True = stalks have collision, False = friction-only mode
    """
    global ENABLE_PROXY_STALKS
    ENABLE_PROXY_STALKS = enabled
    print(f"ENABLE_PROXY_STALKS set to: {enabled}")


def get_effective_friction():
    """
    Calculate and return the effective friction when robot walks on grass.
    
    Returns:
        dict: Static and dynamic effective friction values
    """
    eff_static = (GRASS_MATERIAL_CONFIG["static_friction"] + 
                  ROBOT_FOOT_MATERIAL["static_friction"]) / 2
    eff_dynamic = (GRASS_MATERIAL_CONFIG["dynamic_friction"] + 
                   ROBOT_FOOT_MATERIAL["dynamic_friction"]) / 2
    
    return {
        "effective_static": eff_static,
        "effective_dynamic": eff_dynamic,
        "grass_static": GRASS_MATERIAL_CONFIG["static_friction"],
        "grass_dynamic": GRASS_MATERIAL_CONFIG["dynamic_friction"],
        "robot_static": ROBOT_FOOT_MATERIAL["static_friction"],
        "robot_dynamic": ROBOT_FOOT_MATERIAL["dynamic_friction"],
    }


# =============================================================================
# ISAAC SIM UI STEPS (for documentation)
# =============================================================================

UI_STEPS = """
================================================================================
ISAAC SIM UI STEPS - Manual Grass Physics Setup
================================================================================

A) CREATE/EDIT PHYSX GRASS MATERIAL
-----------------------------------
1. Window > Physics > Physics Inspector
2. Click "Physics Materials" in left panel
3. Click "+" to add new material, name it "GrassMaterial"
4. Set properties:
   - Static Friction: 0.20
   - Dynamic Friction: 0.15
   - Restitution: 0.05
5. In "PhysX Material" section:
   - Friction Combine Mode: Average
   - Restitution Combine Mode: Min

B) APPLY MATERIAL TO GROUND
---------------------------
1. Select ground prim in Stage window (e.g., /World/Ground)
2. In Property panel, find "Physics > Material"
3. Click material slot, select "GrassMaterial"
4. Verify collision is enabled:
   - Check "Physics > Collision" section exists
   - If not: Right-click prim > Add > Physics > Collision

C) DISABLE STALK SELF-COLLISION
-------------------------------
1. Create collision group:
   - Create > Physics > Collision Group
   - Name: "StalkCollisionGroup"
2. In Property panel for collision group:
   - Filtered Groups > Add "StalkCollisionGroup" (itself)
3. Add stalks to group:
   - Select all stalk prims
   - In Physics Inspector, drag to "StalkCollisionGroup"

D) THIN STALK COUNT / DISABLE EXTRAS
------------------------------------
1. Select stalk prims to disable (keep ~900 for 1800 ft² at 0.5/ft²)
2. In Property panel for selected prims:
   - Physics > Collision > Collision Enabled: OFF
3. Or delete extra prims entirely

E) CONFIGURE KEPT STALKS AS KINEMATIC
-------------------------------------
1. Select remaining stalk prims
2. In Property panel:
   - Physics > Rigid Body > Kinematic: ON
3. Apply GrassMaterial to stalks:
   - Physics > Material > Select "GrassMaterial"

================================================================================
"""


if __name__ == "__main__":
    print(UI_STEPS)
    print("\nEffective friction calculation:")
    friction = get_effective_friction()
    for key, value in friction.items():
        print(f"  {key}: {value:.3f}")
