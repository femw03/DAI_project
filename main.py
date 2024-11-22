import sys
import time

from loguru import logger

from DAI.cv import ComputerVisionModuleImp
from DAI.interfaces.interfaces import (
    CarlaData,
    CarlaFeatures,
    CruiseControlAgent,
)
from DAI.simulator import World
from DAI.visuals import Visuals


class MockCruiseControlAgent(CruiseControlAgent):
    def get_action(self, state: CarlaFeatures) -> float:
        return 1


logger.remove()
logger.add(sys.stderr, level="INFO")


visuals = Visuals(640, 480, 30)

cv = ComputerVisionModuleImp()
agent = MockCruiseControlAgent()

world = World(
    view_height=visuals.height,
    view_width=visuals.width,
)


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
    features = cv.process_data(data)
    visuals.detected_objects = features.objects
    action = agent.get_action(features)
    world.set_speed(action)
    world.await_next_tick()


world.join()
visuals.join()
