import os
import sys
sys.path.append(os.getcwd())

from DAI.simulator import CarlaWorld
from DAI.visuals import Visuals
from DAI.interfaces import CarlaData

# Visual setup (screen size, framerate, etc.)
visuals = Visuals(1280, 720, 30)

def set_view_data(data: CarlaData) -> None:
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

def setup_carla() -> CarlaWorld:
    """
    Function to set up the Carla simulation environment.
    Returns the CarlaWorld instance.
    """
    # Initialize CarlaWorld and link to visuals
    world = CarlaWorld(view_height=visuals.height, view_width=visuals.width)
    world.add_listener(set_view_data)

    # Visuals will call stop() when they are done
    visuals.on_stop = stop(world)

    # Start Carla world and visuals
    world.start()
    visuals.start()
    world.await_next_tick()

    return world
