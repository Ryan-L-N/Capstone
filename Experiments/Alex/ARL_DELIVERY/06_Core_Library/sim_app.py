"""
SimulationApp factory - must be called before ANY other Isaac Sim imports.

This is the one module that MUST be imported and called first in every script.
The isaacsim import happens lazily inside the function to avoid circular imports.

Usage:
    from core.sim_app import create_simulation
    app = create_simulation(headless=False, width=1920, height=1080)

    # NOW you can safely import omni.isaac modules
    from omni.isaac.core import World
"""

_simulation_app = None


def create_simulation(headless=False, width=1280, height=720,
                      renderer="RayTracedLighting", anti_aliasing=0, **kwargs):
    """
    Create and return a SimulationApp singleton.

    Args:
        headless: Run without GUI window
        width: Window width in pixels
        height: Window height in pixels
        renderer: "RayTracedLighting" or "PathTracing"
        anti_aliasing: 0=disabled (faster), 1-3 for quality levels
        **kwargs: Additional config passed to SimulationApp

    Returns:
        SimulationApp instance
    """
    global _simulation_app
    if _simulation_app is not None:
        return _simulation_app

    from isaacsim import SimulationApp

    config = {
        "headless": headless,
        "width": width,
        "height": height,
        "window_width": width,
        "window_height": height,
        "renderer": renderer,
        "anti_aliasing": anti_aliasing,
    }
    config.update(kwargs)

    print("Starting Isaac Sim...")
    _simulation_app = SimulationApp(config)
    print("Isaac Sim started successfully!")
    return _simulation_app


def get_simulation_app():
    """
    Get the existing SimulationApp instance.
    Raises RuntimeError if create_simulation() hasn't been called yet.
    """
    if _simulation_app is None:
        raise RuntimeError(
            "SimulationApp not created yet. Call create_simulation() first."
        )
    return _simulation_app
