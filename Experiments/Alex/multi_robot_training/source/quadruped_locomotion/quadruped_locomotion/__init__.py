"""Quadruped locomotion training — Isaac Lab extension package.

Importing this package triggers gym registration for all task environments.
"""

import os

from isaaclab.utils import import_packages

# Auto-discover and register all tasks in the tasks/ subdirectory
import_packages(
    "quadruped_locomotion.tasks",
    os.path.join(os.path.dirname(__file__), "tasks"),
)
