from __future__ import annotations

from loguru import logger
import time
import os
import shutil
import sys

from ...interfaces import CarlaData
from ...cv import ComputerVisionModuleImp
from ...simulator import CarlaWorld
from ...utils import timeit
from ...visuals import Visuals

"""
An example of client-side bounding boxes with basic car controls.

Controls:
Welcome to CARLA for Getting Bounding Box Data.
Use WASD keys for control.
    C            : Capture Data
    l            : Loop Capture Start
    L            : Loop Capture End

    ESC          : quit
"""
try:
    import pygame
    from pygame.locals import K_ESCAPE      # ESC       : quit
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_c           # C         : capture image
    from pygame.locals import K_l           # L         : loop capture
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

# Directories to create/empty
dirs = ['custom_data/', 'SegmentationImage/']

for directory in dirs:
    if os.path.exists(directory):
        shutil.rmtree(directory)
        logger.info("Succefully removed all contents in the directory")
    os.makedirs(directory)
    logger.info("Succefully created the (empty) directory")


class KeyControl():
    def control(self) -> bool:
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """
        keys = pygame.key.get_pressed()

        if keys[K_ESCAPE]:
            logger.info("Pressed ESC key")
            return True 

        if keys[K_c]:
            logger.info("Pressed C key - capture image")
            self.screen_capture = self.screen_capture + 1
        else:
            self.screen_capture = 0

        if keys[K_l]:
            logger.info("Pressed l key - start capture loop")
            self.loop_state = True

        if keys[K_l] and (pygame.key.get_mods() & pygame.KMOD_SHIFT):
            logger.info("Pressed L key - end capture loop")
            self.loop_state = False

        return False


logger.remove()
logger.add(sys.stderr, level="INFO")


visuals = Visuals(1280, 720, 30)

cv = ComputerVisionModuleImp()
keyControl = KeyControl()

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
    if keyControl.control():
        logger.info("Key control input detected")
        continue
    
    data = world.data
    logger.info("Fetching simulation data from CarlaWorld")

    if data is None:
        time.sleep(0)   # yield thread
        continue        # refetch data
    
    features, process_time = timeit(lambda: cv.process_data(data))
    visuals.detected_features = features
    visuals.process_time = process_time
    world.await_next_tick()
    
    if len(world.pedestrians) > 0:
        walker = world.pedestrians[0]
        logger.info(f"{walker.location}, {walker.velocity},{walker.state}")

world.join()
visuals.join()