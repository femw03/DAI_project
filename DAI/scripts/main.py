"""
This file contains the main functionality logic implementation.
This means that running this script will connect to the carla environment,
spawn the cars, set up a visuals window etc...
"""

import argparse
import sys
import time

from loguru import logger

from ..cv import ComputerVisionModuleImp
from ..environment import SACAgent, SimpleFeatureExtractor
from ..interfaces import (
    CarlaData,
)
from ..simulator import CarlaWorld, wrappers
from ..simulator.extract import (
    get_affecting_traffic_lightV2,
    get_stop_point,
    has_collided,
    has_completed_navigation,
)
from ..utils import timeit
from ..visuals import ObjectDTO, Visuals

logger.remove()
logger.add(sys.stderr, level="INFO")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--lead",
    action="store_true",
    default=False,
    help="Always have a lead vehicle, works best if no other cars are present",
)
parser.add_argument("--cars", default=10, help="Amount of cars in the world")
parser.add_argument("--pedestrians", default=50, help="Amount of pedestrians")
parser.add_argument(
    "--cheat",
    action="store_true",
    default=False,
    help="Use perfect traffic light information",
)
args = parser.parse_args()
visuals = Visuals(1280, 720, 30)

cv = ComputerVisionModuleImp()
agent = SACAgent(weights="./DAI/environment/models/agent_best.zip")
feature_extractor = SimpleFeatureExtractor(
    visuals.width,
    visuals.margin,
    visuals.correction_factor,
    visuals.boost_factor,
    traffic_light_distance_bias=20,
)
world = CarlaWorld(
    view_height=visuals.height,
    view_width=visuals.width,
    cars=args.cars,
    walkers=args.pedestrians,
    has_lead_car=args.lead,
    tickrate=10,
)
should_cheat = args.cheat
traffic_lights = wrappers.CarlaTrafficLight.all(world.world)


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
world.await_next_tick()
route = [wp for wp, _ in world.local_planner.get_plan()]

while is_running:
    data = world.data
    if data is None:
        time.sleep(0)  # yield thread
        continue  # refetch data
    observation, process_time = timeit(lambda: cv.process_data(data))
    feature_extractor.boost_factor = visuals.boost_factor
    feature_extractor.correction_factor = visuals.correction_factor
    feature_extractor.margin = visuals.margin
    features = feature_extractor.extract(observation)
    vehicle_in_front = feature_extractor.vehicle_in_front
    objects = [
        ObjectDTO.from_object(obj, is_relevant=vehicle_in_front == obj)
        for obj in observation.objects
    ]

    if should_cheat:
        features.max_speed = 30
        current_light = get_affecting_traffic_lightV2(world, traffic_lights, route)
        if current_light is None:
            features.should_stop = False
            features.distance_to_stop = None
        else:
            stop_point = get_stop_point(world, current_light)
            distance_to_stop = (
                stop_point.location.distance_to(world.car.location)
                if stop_point is not None
                else None
            )
            features.should_stop = (
                current_light.state != (wrappers.CarlaTrafficLightState.GREEN)
                and not stop_point.is_passed(world.car.location)
                if stop_point is not None
                else False
            )
            features.distance_to_stop = (
                distance_to_stop if features.should_stop else None
            )
    visuals.detected_objects = objects
    information = {
        "current speed": features.current_speed,
        "max speed": features.max_speed,
        "stop flag": features.should_stop,
        "distance to stop": features.distance_to_stop,
        "car in front": features.is_car_in_front,
        "distance to car": features.distance_to_car_in_front,
        "process time": process_time,
        # "angle": data.angle,
    }
    visuals.information = information
    visuals.angle = data.angle
    action = agent.get_action(features)
    world.set_speed(action)
    if has_collided(world):
        world.reset()
        route = [wp for wp, _ in world.local_planner.get_plan()]
    elif has_completed_navigation(world):
        route = [wp for wp, _ in world.generate_new_route(world.car.location)]
    world.await_next_tick()


world.join()
visuals.join()
