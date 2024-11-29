"""Contains functions to spawn vehicles and actors as well as destroying a list of actors"""

import logging
import random
from typing import List

import carla
from loguru import logger

from .wrappers import (
    CarlaActor,
    CarlaClient,
    CarlaVehicle,
    CarlaWalker,
    DestroyActorCommand,
)


def spawn_vehicles(client: CarlaClient, vehicle_target: int = 10) -> List[CarlaVehicle]:
    """Spawns vehicle_target amount of random auto_piloted cars in the world. If a car is already present it will not be spawned"""
    logger.info(f"attempting to spawn {vehicle_target} cars")
    world = client.world
    assert world is not None, "Make sure you have loaded a world"

    car_blueprints = world.blueprint_library.filter("vehicle.*")
    filtered_car_bps = [bp for bp in car_blueprints if int(bp["number_of_wheels"]) == 4]

    logger.info(f"Found {len(filtered_car_bps)} valid car blueprints")
    logger.debug(f"{filtered_car_bps=}")

    spawn_points = world.map.spawn_points

    logger.info(f"Found {len(spawn_points)} valid spawn points")

    car_batch = []
    for _ in range(vehicle_target):
        blueprint = random.choice(filtered_car_bps)
        while blueprint.tags[0] in [
            "crossbike",
            "low rider",
            "ninja",
            "yzf",
            "century",
            "omafiets",
            "diamondback",
            "carlacola",
        ]:
            logger.debug(f"skipping {blueprint.tags[0]=}")
            blueprint = random.choice(filtered_car_bps)
        if blueprint.contains("color"):
            blueprint["color"] = random.choice(blueprint["color"].recommended_values)
        if blueprint.contains("driver_id"):
            blueprint["driver_id"] = random.choice(
                blueprint["driver_id"].recommended_values
            )
        blueprint["role_name"] = "autopilot"
        car_batch.append((blueprint, random.choice(spawn_points)))

    logger.debug(f"trying {len(car_batch)} valid cars and spots")

    vehicles: List[CarlaVehicle] = []
    for car_to_spawn, at_location in car_batch:
        try:
            vehicle = world.spawn_vehicle(car_to_spawn, at_location)
            vehicle.autopilot = True
            vehicles.append(vehicle)
        except Exception as e:
            logger.error(f"Failed to spawn car: {e}")
    logger.info(f"succesfully spawned {len(vehicles)} cars")
    logger.debug(f"{vehicles=}")
    return vehicles


def spawn_walkers(
    client: CarlaClient, walker_target=50, pedestrian_cross_factor=0.2
) -> List[CarlaWalker]:
    logger.info(f"attempting to spawn {walker_target} walkers")
    world = client.world
    assert world is not None, "Make sure you have loaded a world"

    walker_blueprints = world.blueprint_library.filter("walker.pedestrian.*")
    logger.info(f"Found {len(walker_blueprints)} valid car blueprints")
    logger.debug(f"{walker_blueprints=}")

    walkers: list[CarlaWalker] = []
    world.pedestrian_crossing_factor = pedestrian_cross_factor
    walker_speeds = []
    for _ in range(walker_target):
        spawn_point = carla.Transform()
        location = world.get_random_walker_location()
        spawn_point.location = location.native

        blueprint = random.choice(walker_blueprints)
        # set as not invencible
        if blueprint.contains("is_invincible"):
            blueprint["is_invincible"] = "false"
        # set the max speed
        speed = 0.0
        if blueprint.contains("speed"):
            # pass  # TODO
            # if random.random() > percentagePedestriansRunning:
            #     # walking
            speed: float = float(blueprint["speed"].recommended_values[1])
        try:
            walker = world.spawn_walker(blueprint, spawn_point)
        except Exception as e:
            logger.error(e)
            continue
        walkers.append(walker)
        walker_speeds.append(speed)
    # Update walker locations
    world.tick()

    # Add controller
    for walker in walkers:
        walker_controller_bp = world.blueprint_library.filter("controller.ai.walker")[0]
        walker.add_controller(walker_controller_bp)

    # Update everything again
    world.tick()
    for walker, speed in zip(walkers, walker_speeds):
        controller = walker.controller
        assert controller is not None
        controller.start()
        controller.speed = speed
        controller.go_to_location(world.get_random_walker_location())
    logger.info(f"successfully spawned {len(walkers)} walkers")
    return walkers


def delete_actors(client: CarlaClient, actors: List[CarlaActor]) -> None:
    logger.info(f"deleting {len(actors)}")
    # for actor in actors:
    #     try:
    #         logger.debug(f"Deleting ${actor.actor}")
    #         actor.destroy()
    #     except Exception as e:
    #         logger.error(f"Exception occured while deleting: {actor}: {e}")
    results = client.apply_batch_sync([DestroyActorCommand(actor) for actor in actors])
    for result in results:
        if result.error is not None or len(str(result.error).strip()) > 0:
            logger.warning(result.error)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    client = CarlaClient()
    client.world.world.get_settings().synchronous_mode = True
    vehicles = spawn_vehicles(client)
    walkers = spawn_walkers(client)
    combined = vehicles + walkers
    input("press enter to quit")
    logger.info(f"Destroying {len(combined)}")
    delete_actors(combined)
