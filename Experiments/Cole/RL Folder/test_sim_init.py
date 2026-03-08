"""Test SimulationApp initialization."""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
args, unknown = parser.parse_known_args()

print("=" * 60)
print("INITIALIZING SIMULATION APP")
print("=" * 60)

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

print("[OK] SimulationApp created successfully")

# Now import Isaac modules
from omni.isaac.core import World
print("[OK] Isaac Core imported")

# Create world
world = World()
print("[OK] World created")

world.reset()
print("[OK] World reset completed")

simulation_app.close()
print("[OK] Simulation closed cleanly")
