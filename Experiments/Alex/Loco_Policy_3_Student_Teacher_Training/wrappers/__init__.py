"""Sim-to-real environment wrappers.

ActionDelayWrapper: Simulates 40-60 ms actuator latency (Risk R1)
ObservationDelayWrapper: Simulates 10-20 ms sensor latency (Risk R1)
SensorNoiseWrapper: Adds dropout, IMU drift, spike noise (Risks R5, R10)

Stacking order (outermost first):
    SensorNoise(ActionDelay(ObsDelay(RslRlVecEnvWrapper(env))))
"""

from .action_delay import ActionDelayWrapper
from .observation_delay import ObservationDelayWrapper
from .sensor_noise import SensorNoiseWrapper

__all__ = ["ActionDelayWrapper", "ObservationDelayWrapper", "SensorNoiseWrapper"]
