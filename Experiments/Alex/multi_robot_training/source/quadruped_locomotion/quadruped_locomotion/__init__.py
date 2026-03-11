"""Quadruped locomotion training — Isaac Lab extension package.

Importing this package triggers gym registration for all task environments
(when running inside Isaac Sim). Direct submodule imports work without
auto-discovery for standalone scripts.
"""

try:
    from isaaclab_tasks.utils import import_packages
    import_packages("quadruped_locomotion.tasks")
except ImportError:
    # isaaclab_tasks not available — skip auto-discovery.
    # Direct imports (e.g. from quadruped_locomotion.tasks.locomotion.config.spot.env_cfg)
    # still work.
    pass
