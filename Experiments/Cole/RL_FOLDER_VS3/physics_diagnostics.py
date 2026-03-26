#!/usr/bin/env python3
"""
Physics Engine Diagnostic Tool for Isaac Sim
Queries and adjusts physics constraints, contact properties, and engine settings
"""

import argparse
from omni.isaac.kit import AppLauncher

# Default launcher settings
launch_config = {
    "headless": True,
    "launcher_args": ["--no-window"],
}

app_launcher = AppLauncher(**launch_config)
app = app_launcher.app

# Now imports can happen
from omni.isaac.core import World
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.nucleus import get_assets_root_path
from pxr import UsdPhysics, Gf, PhysxSchema
import numpy as np


class PhysicsEngineDiagnostics:
    """Query and adjust Isaac Sim physics engine parameters"""
    
    def __init__(self):
        """Initialize diagnostics world"""
        self.world = World(backend="torch", device="cpu")
        self.world.scene.add_default_ground_plane()
        
    def print_physics_scene_settings(self):
        """Print current physics scene configuration"""
        print("\n" + "="*60)
        print("ISAAC SIM PHYSICS SCENE SETTINGS")
        print("="*60)
        
        # Get stage and physics scene
        from omni.isaac.core.utils.stage import get_current_stage
        stage = get_current_stage()
        
        # Find PhysicsScene
        physics_paths = stage.FindPrims(primPath="/", primType="UsdPhysics.PhysicsScene")
        
        if physics_paths:
            for prim_path in physics_paths:
                physics_prim = stage.GetPrimAtPath(str(prim_path.GetPath()))
                print(f"\nPhysics Scene: {prim_path.GetPath()}")
                
                # Get physics attributes
                gravity_attr = physics_prim.GetAttribute("physics:gravity")
                if gravity_attr:
                    gravity = gravity_attr.Get()
                    print(f"  Gravity: {gravity}")
                
                # PhysX-specific settings
                physx_scene = PhysxSchema.PhysxSceneAPI(physics_prim)
                if physx_scene:
                    print(f"  PhysX Settings:")
                    
                    # Solver settings
                    if physx_scene.GetSolverTypeAttr():
                        print(f"    Solver Type: {physx_scene.GetSolverTypeAttr().Get()}")
                    if physx_scene.GetTimeStepsPerSecondAttr():
                        print(f"    Timestepping (Hz): {physx_scene.GetTimeStepsPerSecondAttr().Get()}")
                    if physx_scene.GetSubStepsAttr():
                        print(f"    Substeps: {physx_scene.GetSubStepsAttr().Get()}")
                    
                    # Broadphase settings
                    if physx_scene.GetBroadphaseTypeAttr():
                        print(f"    Broadphase Type: {physx_scene.GetBroadphaseTypeAttr().Get()}")
                    
                    # Constraint settings
                    if physx_scene.GetMaxDepenetrationVelocityAttr():
                        print(f"    Max Depenetration Velocity: {physx_scene.GetMaxDepenetrationVelocityAttr().Get()}")
                    if physx_scene.GetContactOffsetAttr():
                        print(f"    Contact Offset: {physx_scene.GetContactOffsetAttr().Get()}")
                    if physx_scene.GetRestOffsetAttr():
                        print(f"    Rest Offset: {physx_scene.GetRestOffsetAttr().Get()}")
                    
                    # GPU settings
                    if physx_scene.GetEnableGPUDynamicsAttr():
                        print(f"    GPU Dynamics Enabled: {physx_scene.GetEnableGPUDynamicsAttr().Get()}")
                    if physx_scene.GetGPUFoundRigidsAttr():
                        print(f"    GPU Found Rigids: {physx_scene.GetGPUFoundRigidsAttr().Get()}")
        else:
            print("No physics scene found!")
    
    def print_gravity_settings(self):
        """Print and query gravity settings"""
        print("\n" + "="*60)
        print("GRAVITY SETTINGS")
        print("="*60)
        
        from omni.isaac.core.utils.stage import get_current_stage
        stage = get_current_stage()
        
        physics_paths = stage.FindPrims(primPath="/", primType="UsdPhysics.PhysicsScene")
        
        for prim_path in physics_paths:
            physics_prim = stage.GetPrimAtPath(str(prim_path.GetPath()))
            gravity_attr = physics_prim.GetAttribute("physics:gravity")
            if gravity_attr:
                gravity = gravity_attr.Get()
                print(f"\nCurrent Gravity Vector: {gravity} m/s²")
                print(f"Magnitude: {np.linalg.norm(gravity):.6f} m/s²")
                
                # Standard Earth gravity
                standard_gravity = 9.81
                actual_magnitude = np.linalg.norm(gravity)
                print(f"Difference from Earth gravity ({standard_gravity}): {abs(actual_magnitude - standard_gravity):.6f} m/s²")
    
    def print_contact_properties(self):
        """Print default contact properties"""
        print("\n" + "="*60)
        print("CONTACT PROPERTIES")
        print("="*60)
        
        from omni.isaac.core.utils.stage import get_current_stage
        stage = get_current_stage()
        
        physics_paths = stage.FindPrims(primPath="/", primType="UsdPhysics.PhysicsScene")
        
        for prim_path in physics_paths:
            physics_prim = stage.GetPrimAtPath(str(prim_path.GetPath()))
            physx_scene = PhysxSchema.PhysxSceneAPI(physics_prim)
            
            if physx_scene:
                print("\nDefault Contact Properties:")
                
                contact_offset = physx_scene.GetContactOffsetAttr()
                rest_offset = physx_scene.GetRestOffsetAttr()
                max_depenetration = physx_scene.GetMaxDepenetrationVelocityAttr()
                
                if contact_offset:
                    print(f"  Contact Offset: {contact_offset.Get()} m")
                if rest_offset:
                    print(f"  Rest Offset: {rest_offset.Get()} m")
                if max_depenetration:
                    print(f"  Max Depenetration Velocity: {max_depenetration.Get()} m/s")
    
    def print_timestep_settings(self):
        """Print simulation timestep configuration"""
        print("\n" + "="*60)
        print("TIMESTEP CONFIGURATION")
        print("="*60)
        
        from omni.isaac.core.utils.stage import get_current_stage
        stage = get_current_stage()
        
        physics_paths = stage.FindPrims(primPath="/", primType="UsdPhysics.PhysicsScene")
        
        for prim_path in physics_paths:
            physics_prim = stage.GetPrimAtPath(str(prim_path.GetPath()))
            physx_scene = PhysxSchema.PhysxSceneAPI(physics_prim)
            
            if physx_scene:
                if physx_scene.GetTimeStepsPerSecondAttr():
                    hz = physx_scene.GetTimeStepsPerSecondAttr().Get()
                    dt = 1.0 / hz if hz > 0 else 0
                    print(f"\nSimulation Frequency: {hz} Hz")
                    print(f"Timestep (dt): {dt:.6f} seconds ({dt*1000:.3f} ms)")
                
                if physx_scene.GetSubStepsAttr():
                    substeps = physx_scene.GetSubStepsAttr().Get()
                    print(f"Substeps per Frame: {substeps}")
    
    def print_constraint_settings(self):
        """Print constraint solving parameters"""
        print("\n" + "="*60)
        print("CONSTRAINT SOLVING SETTINGS")
        print("="*60)
        
        from omni.isaac.core.utils.stage import get_current_stage
        stage = get_current_stage()
        
        physics_paths = stage.FindPrims(primPath="/", primType="UsdPhysics.PhysicsScene")
        
        for prim_path in physics_paths:
            physics_prim = stage.GetPrimAtPath(str(prim_path.GetPath()))
            physx_scene = PhysxSchema.PhysxSceneAPI(physics_prim)
            
            if physx_scene:
                print("\nConstraint Solver Parameters:")
                
                # Solver iterations
                if hasattr(physx_scene, "GetSolverIterationCountAttr"):
                    iterations = physx_scene.GetSolverIterationCountAttr()
                    if iterations:
                        print(f"  Solver Iterations: {iterations.Get()}")
                
                # Position Iterations
                if hasattr(physx_scene, "GetPositionIterationCountAttr"):
                    pos_iter = physx_scene.GetPositionIterationCountAttr()
                    if pos_iter:
                        print(f"  Position Iterations: {pos_iter.Get()}")
                
                # Velocity Iterations
                if hasattr(physx_scene, "GetVelocityIterationCountAttr"):
                    vel_iter = physx_scene.GetVelocityIterationCountAttr()
                    if vel_iter:
                        print(f"  Velocity Iterations: {vel_iter.Get()}")
    
    def adjust_contact_properties(self, contact_offset=None, rest_offset=None, 
                                  max_depenetration=None):
        """Adjust contact properties"""
        print("\n" + "="*60)
        print("ADJUSTING CONTACT PROPERTIES")
        print("="*60)
        
        from omni.isaac.core.utils.stage import get_current_stage
        stage = get_current_stage()
        
        physics_paths = stage.FindPrims(primPath="/", primType="UsdPhysics.PhysicsScene")
        
        for prim_path in physics_paths:
            physics_prim = stage.GetPrimAtPath(str(prim_path.GetPath()))
            physx_scene = PhysxSchema.PhysxSceneAPI(physics_prim)
            
            if physx_scene:
                if contact_offset is not None:
                    physx_scene.CreateContactOffsetAttr().Set(contact_offset)
                    print(f"✓ Contact Offset set to: {contact_offset} m")
                
                if rest_offset is not None:
                    physx_scene.CreateRestOffsetAttr().Set(rest_offset)
                    print(f"✓ Rest Offset set to: {rest_offset} m")
                
                if max_depenetration is not None:
                    physx_scene.CreateMaxDepenetrationVelocityAttr().Set(max_depenetration)
                    print(f"✓ Max Depenetration Velocity set to: {max_depenetration} m/s")
    
    def print_full_diagnostic(self):
        """Print full system diagnostic"""
        print("\n\n")
        print("#" * 60)
        print("# ISAAC SIM PHYSICS ENGINE DIAGNOSTIC REPORT")
        print("#" * 60)
        
        self.print_physics_scene_settings()
        self.print_gravity_settings()
        self.print_timestep_settings()
        self.print_contact_properties()
        self.print_constraint_settings()
        
        print("\n" + "#" * 60)
        print("# DIAGNOSTIC COMPLETE")
        print("#" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Isaac Sim Physics Engine Diagnostics")
    parser.add_argument("--diagnostic", action="store_true", default=False,
                        help="Run full physics diagnostic")
    parser.add_argument("--gravity", action="store_true", default=False,
                        help="Print gravity settings")
    parser.add_argument("--contact", action="store_true", default=False,
                        help="Print contact properties")
    parser.add_argument("--timestep", action="store_true", default=False,
                        help="Print timestep configuration")
    parser.add_argument("--constraints", action="store_true", default=False,
                        help="Print constraint solving settings")
    parser.add_argument("--adjust-contact-offset", type=float, default=None,
                        help="Adjust contact offset (meters)")
    parser.add_argument("--adjust-rest-offset", type=float, default=None,
                        help="Adjust rest offset (meters)")
    parser.add_argument("--adjust-depenetration", type=float, default=None,
                        help="Adjust max depenetration velocity (m/s)")
    
    args = parser.parse_args()
    
    # Initialize diagnostics
    diag = PhysicsEngineDiagnostics()
    
    # Determine what to run
    if args.diagnostic or (not any([args.gravity, args.contact, args.timestep, 
                                     args.constraints, args.adjust_contact_offset,
                                     args.adjust_rest_offset, args.adjust_depenetration])):
        # Full diagnostic by default
        diag.print_full_diagnostic()
    else:
        if args.gravity:
            diag.print_gravity_settings()
        if args.contact:
            diag.print_contact_properties()
        if args.timestep:
            diag.print_timestep_settings()
        if args.constraints:
            diag.print_constraint_settings()
        
        # Apply adjustments
        if any([args.adjust_contact_offset, args.adjust_rest_offset, 
                args.adjust_depenetration]):
            diag.adjust_contact_properties(
                contact_offset=args.adjust_contact_offset,
                rest_offset=args.adjust_rest_offset,
                max_depenetration=args.adjust_depenetration
            )


if __name__ == "__main__":
    main()
