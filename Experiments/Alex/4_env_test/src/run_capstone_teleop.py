"""Xbox controller teleop for manual single-robot walkthrough.

Supports gait switching (FLAT <-> ROUGH), FPV camera, and 4 drive modes.

Controls:
  Xbox:
    Left Stick Y:  Forward/backward
    Left Stick X:  Turn left/right
    A:             Cycle drive mode (MANUAL -> SMOOTH -> PATROL -> AUTO-NAV)
    B:             Toggle selfright
    Y:             Reset position
    LB:            Toggle FPV camera
    RB:            Cycle gait (FLAT <-> ROUGH)
    D-Pad:         Speed multiplier
    Back:          E-stop

  Keyboard:
    W/S:           Forward/backward
    A/D:           Turn left/right
    G:             Cycle gait
    M:             Toggle FPV camera
    SHIFT:         Cycle drive mode
    X:             Toggle selfright
    R:             Reset
    SPACE:         E-stop
    ESC:           Exit

Usage:
    ./isaaclab.sh -p src/run_capstone_teleop.py --env friction --device xbox

Reuses patterns from:
- ARL_DELIVERY/04_Teleop_System/spot_teleop.py (Xbox mapping, drive modes, FPV)
- ARL_DELIVERY/02_Obstacle_Course/spot_obstacle_course.py (gait switching)
- ARL_DELIVERY/03_Rough_Terrain_Policy/spot_rough_terrain_policy.py (policy hot-swap)
"""

# TODO: Implementation
# 1. Parse CLI args (--env, --device)
# 2. Initialize SimulationApp (rendered, single robot)
# 3. Build environment via envs.build_environment()
# 4. Load both flat and rough policies
# 5. Setup Xbox controller (pygame XInput, 12% deadzone)
# 6. Setup keyboard fallback
# 7. Main loop:
#    - Read controller input
#    - Apply drive mode scaling
#    - Handle gait switching (RB/G): save PhysX state, swap policy, 0.5s stabilize
#    - Handle FPV toggle (LB/M): switch viewport camera
#    - Compute actions from active policy
#    - Step simulation


def main():
    raise NotImplementedError("TODO: Implement teleop entry point")


if __name__ == "__main__":
    main()
