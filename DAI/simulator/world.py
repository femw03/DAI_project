from __future__ import annotations

import random
from threading import Thread
from typing import Callable, List, Optional

import carla
import numpy as np
from loguru import logger
from pygame.time import Clock

from ..interfaces import CarlaBridge, Image, Lidar
from .numpy_image import NumpyImage, NumpyLidar
from .spawner import delete_actors, spawn_vehicles, spawn_walkers
from .wrappers import (
    CarlaActor,
    CarlaClient,
    CarlaDepthBlueprint,
    CarlaImage,
    CarlaRGBBlueprint,
    CarlaVehicle,
)


class World(Thread, CarlaBridge):
    def __init__(
        self,
        on_image_received: Callable[[Image, Lidar]],
        on_rgb_received: Callable[[np.ndarray], None],
        tickrate=30,
        host="127.0.0.1",
        port=2000,
        view_width=1920 // 2,
        view_height=1080 // 2,
        view_FOV=90,
        walkers=50,
        cars=10,
    ) -> None:
        CarlaBridge.__init__(self, on_image_received=on_image_received)
        Thread.__init__(self)
        self.tickrate = tickrate
        self.view_width = view_width
        self.view_height = view_height
        self.view_FOV = view_FOV
        self.host = host
        self.port = port
        self.number_of_walkers = walkers
        self.number_of_cars = cars
        self.on_rgb_received = on_rgb_received

        self.rgb_image: Optional[np.ndarray] = None
        self.depth_image: Optional[np.ndarray] = None
        self.client = CarlaClient(port=port)
        self.car: Optional[CarlaVehicle] = None
        self.all_actors: List[CarlaActor] = []
        self.sentRGB: Optional[Image] = None
        self.sentDepth: Optional[Image] = None
        self.is_processing = False

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
            self.on_rgb_received(image.numpy_image)

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
            image.native.save_to_disk("test.data", carla.ColorConverter.Raw)
            # depth_data = image.to_depth()
            # self.depth_image = np.stack(
            #     [depth_data.reshape([image.width, image.height])] * 3, axis=-1
            # )
            self.depth_image = np.empty((self.view_width, self.view_height))

        self.depth_camera.listen(save_depth_image)

    def setup(self):
        world = self.client.world
        world.delta_seconds = 1 / self.tickrate
        world.synchronous_mode = True

        self.setup_car()

        self.all_actors.extend(spawn_vehicles(self.client, self.number_of_cars))
        self.all_actors.extend(spawn_walkers(self.client, self.number_of_walkers))
        self.loop_running = True

    def run(self):
        """
        Main program loop.
        """
        try:
            self.setup()
            framecount = 0
            clock = Clock()
            while self.loop_running:
                logger.debug(f"frame: {framecount}   ")
                clock.tick(self.tickrate)
                self.client.world.tick()

                if (
                    self.sentRGB is None
                    and self.sentDepth is None
                    and self.rgb_image is not None
                    and self.depth_image is not None
                ):
                    logger.info("Sending an observation")
                    self.add_image(
                        image=NumpyImage(self.rgb_image),
                        lidar=NumpyLidar(
                            self.depth_image
                        ),  # TODO replace with depth data
                    )

        finally:
            self.client.world.synchronous_mode = False
            delete_actors(self.client, self.all_actors)

    def _add_image(self, image: Image, lidar: Lidar) -> None:
        self.sentDepth = lidar
        self.sentRGB = image

    def set_speed(speed: float) -> None:
        pass  # TODO control speed of car.

    def stop(self) -> None:
        logger.info("Stopping simulation")
        self.loop_running = False


def convert_image(image_array, width, height) -> np.ndarray:
    array = np.frombuffer(image_array, dtype=np.dtype("uint8"))
    array = np.reshape(array, (height, width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array
