"""Nav Locomotion — Phase C visual terrain navigation for Spot quadruped.

Standalone Isaac Lab extension that trains a depth-camera navigation policy
on top of a frozen Phase B locomotion policy. The nav policy outputs velocity
commands [vx, vy, wz] at 10 Hz, which a frozen loco policy converts to
12-dim joint positions at 50 Hz.

Install:
    pip install -e source/nav_locomotion/

Verify:
    python -c "import nav_locomotion; print('OK')"
"""

import os

# Auto-discover and register gym environments via Isaac Lab's import system.
# Graceful fallback if isaaclab_tasks not available (for standalone scripts).
try:
    from isaaclab_tasks.utils import import_packages
    import_packages(__name__, os.path.dirname(__file__))
except ImportError:
    pass

# Explicit registration fallback — import_packages may not find nested configs
try:
    import nav_locomotion.tasks.navigation.config.spot  # noqa: F401 — triggers gym.register
except Exception:
    pass
