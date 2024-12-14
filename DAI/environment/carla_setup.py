import os
import sys

sys.path.append(os.getcwd())

from ..interfaces import CarlaData
from ..simulator import CarlaWorld
from ..visuals import Visuals

# Visual setup (screen size, framerate, etc.)


def set_view_data(data: CarlaData, visuals: Visuals) -> None:
    """
    Function to set the visual data from Carla.
    """
    visuals.depth_image = data.lidar_data.get_lidar_bytes()
    visuals.rgb_image = data.rgb_image.get_image_bytes()


def stop(world: CarlaWorld) -> None:
    """
    Function to stop Carla simulation and set the global running state to False.
    """
    global is_running
    world.stop()
    is_running = False


def setup_carla(visuals: Visuals) -> CarlaWorld:
    """
    Function to set up the Carla simulation environment.
    Returns the CarlaWorld instance.
    """
    # Initialize CarlaWorld and link to visuals
    world = CarlaWorld(
        view_height=visuals.height,
        view_width=visuals.width,
        walkers=0,
        cars=0,
        has_lead_car=True,
    )
    world.add_listener(lambda data: set_view_data(data, visuals))

    # Visuals will call stop() when they are done
    visuals.on_quit = lambda: stop(world)

    # Start Carla world and visuals
    world.start()
    visuals.start()
    world.await_next_tick()

    return world
