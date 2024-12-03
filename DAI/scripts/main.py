"""
This file contains the main functionality logic implementation.
This means that running this script will connect to the carla environment,
spawn the cars, set up a visuals window etc...
"""

import sys
import time

from loguru import logger

from ..cv import ComputerVisionModuleImp
from ..interfaces import (
    CarlaData,
    CarlaFeatures,
    CruiseControlAgent,
)
from ..simulator import CarlaWorld
from ..utils import timeit
from ..visuals import Visuals


class MockCruiseControlAgent(CruiseControlAgent):
    def get_action(self, state: CarlaFeatures) -> float:
        return 1


logger.remove()
logger.add(sys.stderr, level="INFO")


visuals = Visuals(1280, 720, 30)

cv = ComputerVisionModuleImp()
agent = MockCruiseControlAgent()

world = CarlaWorld(view_height=visuals.height, view_width=visuals.width)


def set_view_data(data: CarlaData) -> None:
    visuals.depth_image = data.lidar_data.get_lidar_bytes()
    visuals.rgb_image = data.rgb_image.get_image_bytes()


is_running = True


def stop() -> None:
    global is_running
    world.stop()
    is_running = False


world.add_listener(set_view_data)

visuals.on_quit = stop

world.start()
visuals.start()

while is_running:
    data = world.data
    if data is None:
        time.sleep(0)  # yield thread
        continue  # refetch data
    features, process_time = timeit(lambda: cv.process_data(data))
    visuals.detected_features = features
    visuals.process_time = process_time
    action = agent.get_action(features)
    world.set_speed(action)
    world.await_next_tick()
    if len(world.pedestrians) > 0:
        walker = world.pedestrians[0]
        logger.info(f"{walker.location}, {walker.velocity},{walker.state}")


world.join()
visuals.join()