from __future__ import annotations
import logging
import random
from typing import Optional, List
import numpy as np
import pygame
import pygame.locals
from .spawner import spawn_vehicles, spawn_walkers, delete_actors
from .wrappers import (
    CarlaClient,
    CarlaVehicle,
    CarlaRGBBlueprint,
    CarlaImage,
    CarlaActor,
    CarlaDepthBlueprint,
)

import os

logger = logging.getLogger(__name__)


class World:
    def __init__(
        self,
        framerate=30,
        host="127.0.0.1",
        port=2000,
        view_width=1920 // 2,
        view_height=1080 // 2,
        view_FOV=90,
        walkers=50,
        cars=10,
    ) -> None:
        self.framerate = 30
        self.view_width = view_width
        self.view_height = view_height
        self.view_FOV = view_FOV
        self.host = host
        self.port = port
        self.number_of_walkers = walkers
        self.number_of_cars = cars

        self.rgb_image: Optional[np.ndarray] = None
        self.depth_image: Optional[np.ndarray] = None
        self.client = CarlaClient(port=2000)
        self.car: Optional[CarlaVehicle] = None
        self.all_actors: List[CarlaActor] = []

    def setup_car(self):
        world = self.client.world
        car_bp = world.blueprint_library.filter("vehicle.*")[0]
        location = random.choice(world.map.spawn_points)
        logger.info(f"Spawning {car_bp} at {location}")
        self.car = world.spawn_vehicle(car_bp, location)

        rgb_camera_bp = CarlaRGBBlueprint.from_blueprint(
            world.blueprint_library.filter("sensor.camera.rgb")[0]
        )
        rgb_camera_bp["image_size_x"] = str(self.view_width)
        rgb_camera_bp["image_size_y"] = str(self.view_height)
        rgb_camera_bp["fov"] = str(self.view_FOV)

        self.rgb_camera = self.car.add_camera(rgb_camera_bp)

        def save_rgb_image(image: CarlaImage):
            logger.debug("received image")
            self.rgb_image = image.numpy_image

        self.rgb_camera.listen(save_rgb_image)  # Actor may not lose scope
        self.all_actors.append(self.car)

        depth_camera_bp = CarlaDepthBlueprint.from_blueprint(
            world.blueprint_library.filter("sensor.camera.depth")[0]
        )
        depth_camera_bp["image_size_x"] = str(self.view_width)
        depth_camera_bp["image_size_y"] = str(self.view_height)
        depth_camera_bp["fov"] = str(self.view_FOV)
        self.depth_camera = self.car.add_camera(depth_camera_bp)

        def save_depth_image(image: CarlaImage):
            depth_data = image.to_depth()
            self.depth_image = np.stack(
                [depth_data.reshape([image.width, image.height])] * 3, axis=-1
            )

        self.depth_camera.listen(save_depth_image)

    def setup(self):
        logger.info("Using display 1")
        os.environ["DISPLAY"] = ":1"
        world = self.client.world
        world.delta_seconds = 1 / self.framerate
        world.synchronous_mode = True
        pygame.init()

        self.setup_car()
        self.display = pygame.display.set_mode(
            (self.view_width, self.view_height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self.clock = pygame.time.Clock()

        self.all_actors.extend(spawn_vehicles(self.client, self.number_of_cars))
        self.all_actors.extend(spawn_walkers(self.client, self.number_of_walkers))
        self.running = True

    def start(self):
        """
        Main program loop.
        """
        try:
            self.setup()
            framecount = 0
            display_rgb = True
            while self.running:
                logger.debug(f"frame: {framecount}   ")
                self.client.world.tick()

                self.capture = True
                self.clock.tick_busy_loop(self.framerate)
                image = self.rgb_image if display_rgb else self.depth_image
                if image is not None:
                    logger.debug(f"blitting image {image}")
                    surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
                    self.display.blit(surface, (0, 0))

                pygame.display.flip()

                pygame.event.pump()
                keys = pygame.key.get_pressed()
                if keys[pygame.locals.K_ESCAPE]:
                    self.running = False
                    break
                if keys[pygame.locals.K_a]:
                    self.car.autopilot = not self.car.autopilot
                if keys[pygame.locals.K_t]:
                    logger.info("switching image")
                    display_rgb = not display_rgb
                framecount += 1

        finally:
            self.client.world.synchronous_mode = False
            delete_actors(self.client, self.all_actors)
            pygame.quit()


def convert_image(image_array, width, height) -> np.ndarray:
    array = np.frombuffer(image_array, dtype=np.dtype("uint8"))
    array = np.reshape(array, (height, width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array
