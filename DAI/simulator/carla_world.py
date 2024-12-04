from __future__ import annotations

import random
from threading import Thread
from typing import List, Optional

import numpy as np
from loguru import logger
from pygame.time import Clock

from ..interfaces import CarlaData, World
from .numpy_image import NumpyImage, NumpyLidar
from .spawner import delete_actors, spawn_vehicles, spawn_walkers
from .wrappers import (
    CarlaActor,
    CarlaClient,
    CarlaCollisionEvent,
    CarlaColorConverter,
    CarlaDepthBlueprint,
    CarlaImage,
    CarlaRGBBlueprint,
    CarlaVehicle,
)


class CarlaWorld(Thread, World):
    """
    Bindings for the Carla simulator. Will spawn the walkers, cars and ego_vehicle on start call.
    The simulator is updated in it's own thread so that a similarity to real time simulation is achieved
    """

    def __init__(
        self,
        tickrate=30,
        host="127.0.0.1",
        port=2000,
        view_width=1920 // 2,
        view_height=1080 // 2,
        view_FOV=90,
        walkers=50,
        cars=10,
    ) -> None:
        World.__init__(self)
        Thread.__init__(self)
        self.tickrate = tickrate
        self.view_width = view_width
        self.view_height = view_height
        self.view_FOV = view_FOV
        self.host = host
        self.port = port
        self.number_of_walkers = walkers
        self.number_of_cars = cars

        self.rgb_image: Optional[np.ndarray] = None
        self.depth_image: Optional[np.ndarray] = None
        self.segm_image: Optional[np.ndarray] = None
        self.collision: Optional[CarlaCollisionEvent] = None
        self.client = CarlaClient(port=port)
        self.car: Optional[CarlaVehicle] = None
        self.all_actors: List[CarlaActor] = []
        self.cars: List[CarlaActor] = []
        self.pedestrians: List[CarlaActor] = []

    def setup_car(self):
        """Spawns the car and attaches the rgb- and depth camera to it and setups their listeners"""
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
            converted = image.convert(CarlaColorConverter.DEPTH())
            numpy_image = converted.numpy_image[..., 0]
            if numpy_image is None:
                return
            self.depth_image = numpy_image
            logger.debug(
                f"Depth image [{self.depth_image.min(), self.depth_image.max()}] shape: {self.depth_image.shape}"
            )

        self.depth_camera.listen(save_depth_image)

        segm_camera_bp = world.blueprint_library.filter(
            "sensor.camera.semantic_segmentation"
        )[0]

        segm_camera_bp["image_size_x"] = str(self.view_width)
        segm_camera_bp["image_size_y"] = str(self.view_height)
        segm_camera_bp["fov"] = str(self.view_FOV)

        self.segm_camera = self.car.add_camera(segm_camera_bp)

        def save_segm_image(image: CarlaImage):
            logger.debug("received image")
            self.segm_image = image.numpy_image

        self.segm_camera.listen(save_segm_image)

        self.collision_detector = self.car.add_colision_detector()

        def save_collision(event: CarlaCollisionEvent) -> None:
            logger.warning("Car has collided")
            self.collision = event

        self.collision_detector.listen(save_collision)
        self.car.autopilot = True

    def setup(self):
        """Spawns the car and external actors"""
        world = self.client.world
        world.delta_seconds = 1 / self.tickrate
        world.synchronous_mode = True
        tm = self.client.get_traffic_manager()
        tm.synchronous_mode = True

        self.setup_car()

        self.cars = spawn_vehicles(self.client, self.number_of_cars)
        self.pedestrians = spawn_walkers(self.client, self.number_of_walkers)
        self.all_actors = [*self.cars, *self.pedestrians]
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
                # self.collision = None # Reset the collision state smartly?
                self.client.world.tick()
                self._notify_tick_listeners()

                if self.rgb_image is not None and self.depth_image is not None:
                    logger.debug("Sending an observation")
                    self._set_data(
                        CarlaData(
                            rgb_image=NumpyImage(self.rgb_image, self.view_FOV),
                            lidar_data=NumpyLidar(
                                self.depth_image,
                                self.view_FOV,
                                converter=lambda x: (x / 255) * 1000,
                            ),
                            current_speed=self.car.velocity.magnitude,
                        )
                    )

            # TODO apply car control

        finally:
            self.client.world.synchronous_mode = False
            delete_actors(self.client, self.all_actors)
            self._notify_tick_listeners()

    def stop(self) -> None:
        logger.info("Stopping simulation")
        self.loop_running = False
