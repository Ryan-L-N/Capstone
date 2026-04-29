"""Navigation MDP components — rewards, observations, and terrain configs.

Note: terrains are NOT imported here to avoid pxr dependency at import time.
Import them directly: from nav_locomotion.tasks.navigation.mdp.terrains import ...
"""

# Rewards and observations are safe to import (no pxr dependency)
# but they import isaaclab types that also need pxr, so defer all.
