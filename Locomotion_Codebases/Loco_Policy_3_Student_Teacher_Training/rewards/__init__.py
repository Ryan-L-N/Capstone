"""New reward terms for sim-to-real hardened training.

motor_power: Penalizes |torque * joint_vel| (energy efficiency, Risk R7)
torque_limit: Penalizes torques exceeding real Spot motor limits (Risk R6)
"""

from .motor_power import motor_power_penalty
from .torque_limit import torque_limit_penalty

__all__ = ["motor_power_penalty", "torque_limit_penalty"]
