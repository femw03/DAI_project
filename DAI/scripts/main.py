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
    CarlaObservation,
    CruiseControlAgent,
)
from ..simulator import CarlaWorld
from ..simulator.extract import (
    find_vehicle_in_front,
    get_current_affecting_light_state,
    get_current_max_speed,
    get_current_speed,
    get_objects,
    get_steering_angle,
)
from ..simulator.tracker import find_next_wp_from
from ..simulator.wrappers import CarlaVector3D
from ..utils import timeit
from ..visuals import ObjectDTO, Visuals


class MockCruiseControlAgent(CruiseControlAgent):
    def get_action(self, state: CarlaObservation) -> float:
        return 0.82


logger.remove()
logger.add(sys.stderr, level="INFO")


visuals = Visuals(1280, 720, 30)

cv = ComputerVisionModuleImp()
agent = MockCruiseControlAgent()

world = CarlaWorld(
    view_height=visuals.height,
    view_width=visuals.width,
    cars=80,
    walkers=0,
    has_lead_car=True,
)


def set_view_data(data: CarlaData) -> None:
    visuals.depth_image = data.lidar_data.get_lidar_bytes()
    visuals.rgb_image = data.rgb_image.get_image_bytes()


is_running = True


def stop() -> None:
    global is_running
    world.stop()
    is_running = False


def pause(is_paused) -> None:
    world.paused = is_paused


world.add_listener(set_view_data)

visuals.on_quit = stop
visuals.on_reset = world.reset
visuals.on_pause = pause

world.start()
visuals.start()

while is_running:
    data = world.data
    if data is None:
        time.sleep(0)  # yield thread
        continue  # refetch data
    features, process_time = timeit(lambda: cv.process_data(data))
    angle = get_steering_angle(world)
    route = [waypoint for waypoint, _ in world.local_planner.get_plan()]
    next_wp_result = find_next_wp_from(route, min_distance=20)
    if next_wp_result is not None:
        angle = CarlaVector3D(world.car.transform.get_forward_vector()).angle_to(
            world.car.location.vector_to(next_wp_result[0].location)
        )
    vehicle_in_front = find_vehicle_in_front(
        angle,
        features.objects,
        width=visuals.width,
        correction_factor=visuals.correction_factor,
        boost_factor=visuals.boost_factor,
        threshold=visuals.margin,
    )
    objects = [
        ObjectDTO.from_object(obj, is_relevant=vehicle_in_front == obj)
        for obj in features.objects
    ]

    visuals.detected_objects = objects
    information = {
        "current speed": features.current_speed,
        "max speed": features.max_speed,
        "stop flag": features.stop_flag,
        "process time": process_time,
        "angle": angle,
    }
    visuals.information = information
    visuals.angle = angle
    action = agent.get_action(features)
    world.set_speed(action)
    world.await_next_tick()
    get_current_speed(world)
    get_current_max_speed(world)
    get_objects(world)
    get_current_affecting_light_state(world)
    if len(world.pedestrians) > 0:
        walker = world.pedestrians[0]
        logger.debug(f"{walker.location}, {walker.velocity},{walker.state}")


world.join()
visuals.join()
