#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""

from typing import List

import carla

import logging
import random
from .wrappers import (
    CarlaClient,
    CarlaVehicle,
    CarlaWalker,
    CarlaActor,
    DestroyActorCommand,
)

logger = logging.getLogger(__name__)


def spawn_vehicles(client: CarlaClient, vehicle_target: int = 10) -> List[CarlaVehicle]:
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

    for i in range(walker_target):
        spawn_point = carla.Transform()
        location = world.get_random_walker_location()
        spawn_point.location = location.native

        blueprint = random.choice(walker_blueprints)
        # set as not invencible
        if blueprint.contains("is_invincible"):
            blueprint["is_invincible"] = "false"
        # set the max speed
        if blueprint.contains("speed"):
            pass  # TODO
            # if random.random() > percentagePedestriansRunning:
            #     # walking
            #     walker_speed.append(
            #         walker_bp.get_attribute("speed").recommended_values[1]
            #     )
        try:
            walker = world.spawn_walker(blueprint, spawn_point)
        except Exception as e:
            logger.error(e)
            continue
        walkers.append(walker)

        walker_controller_bp = world.blueprint_library.filter("controller.ai.walker")[0]
        walker.add_controller(walker_controller_bp)
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
        if result.error is not None:
            logger.warning(result.error)

    # finally:
    #     print("\ndestroying %d vehicles" % len(vehicles_list))
    #     self.client.apply_batch(
    #         [carla.command.DestroyActor(x) for x in vehicles_list]
    #     )

    #     # stop walker controllers (list is [controller, actor, controller, actor ...])
    #     for i in range(0, len(all_id), 2):
    #         all_actors[i].stop()

    #     print("\ndestroying %d walkers" % len(walkers_list))
    #     self.client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

    #     time.sleep(0.5)


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