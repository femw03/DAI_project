#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""

import time

import carla

import logging
import random
import threading
from .wrappers import CarlaClient, SetAutoPiloteCommand, SpawnActorCommand

logger = logging.getLogger(__name__)


def spawn(client: CarlaClient, cars: int = 10, walkers: int = 50):
    vehicles_list = []
    walkers_list = []

    logger.info(f"attempting to spawn {cars} cars and {walkers} walkers")
    world = client.world
    assert world is not None, "Make sure you have loaded a world"

    car_blueprints = world.blueprint_library.filter("vehicle.*")
    walker_blueprints = world.blueprint_library.filter("walker.pedestrian.*")
    filtered_car_bps = [bp for bp in car_blueprints if bp["number_of_wheels"] == 4]

    logger.info(
        f"Found {len(walker_blueprints)} valid walker blueprints and {len(filtered_car_bps)} valid car blueprints"
    )
    logger.debug(f"{walker_blueprints:=}, {filtered_car_bps:=}")

    spawn_points = world.map.spawn_points

    logger.info(f"Found {len(spawn_points)} valid spawn points")

    car_batch = []
    for _ in range(cars):
        blueprint = random.choice(filtered_car_bps)
        if blueprint.contains("color"):
            blueprint["color"] = random.choice(blueprint["color"].recommended_values)
        if blueprint["driver_id"]:
            blueprint["driver_id"] = random.choice(
                blueprint["driver_id"].recommended_values
            )
        blueprint["role_name"] = "autopilot"
        car_batch.append(
            SpawnActorCommand(blueprint, random.choice(spawn_points)).then(
                SetAutoPiloteCommand(True)
            )
        )
    for response in client.apply_batch_sync(car_batch):
        if response.error:
            logger.error(response.error)
        else:
            vehicles_list.append(response.actor_id)
    logger.info(f"succesfully spawned {len(vehicles_list)} cars")


class Spawner(threading.Thread):
    def __init__(
        self, client: carla.Client, vehicles: int = 10, walkers: int = 50, safe=True
    ) -> None:
        super().__init__()
        self.vehicles = vehicles
        self.walkers = walkers
        self.safe = safe
        self.running = False
        self.client = client

    def run(self) -> None:
        self.running = True
        # logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

        vehicles_list = []
        walkers_list = []
        all_id = []
        try:
            world = self.client.get_world()
            blueprints = world.get_blueprint_library().filter("vehicle.*")
            blueprintsWalkers = world.get_blueprint_library().filter(
                "walker.pedestrian.*"
            )

            if self.safe:
                blueprints = [
                    x
                    for x in blueprints
                    if int(x.get_attribute("number_of_wheels")) == 4
                ]
                blueprints = [x for x in blueprints if not x.id.endswith("isetta")]
                blueprints = [x for x in blueprints if not x.id.endswith("carlacola")]

            spawn_points = world.get_map().get_spawn_points()
            number_of_spawn_points = len(spawn_points)

            if self.vehicles < number_of_spawn_points:
                random.shuffle(spawn_points)
            elif self.vehicles > number_of_spawn_points:
                msg = "requested %d self.vehicles, but could only find %d spawn points"
                logging.warning(msg, self.vehicles, number_of_spawn_points)
                self.vehicles = number_of_spawn_points

            # @todo cannot import these directly.
            SpawnActor = carla.command.SpawnActor
            SetAutopilot = carla.command.SetAutopilot
            FutureActor = carla.command.FutureActor

            # --------------
            # Spawn vehicles
            # --------------
            batch = []
            for n, transform in enumerate(spawn_points):
                if n >= self.vehicles:
                    break
                blueprint = random.choice(blueprints)
                if blueprint.tags[0] not in [
                    "crossbike",
                    "low rider",
                    "ninja",
                    "yzf",
                    "century",
                    "omafiets",
                    "diamondback",
                    "carlacola",
                ]:
                    if blueprint.has_attribute("color"):
                        color = random.choice(
                            blueprint.get_attribute("color").recommended_values
                        )
                        blueprint.set_attribute("color", color)
                    if blueprint.has_attribute("driver_id"):
                        driver_id = random.choice(
                            blueprint.get_attribute("driver_id").recommended_values
                        )
                        blueprint.set_attribute("driver_id", driver_id)
                    blueprint.set_attribute("role_name", "autopilot")
                    batch.append(
                        SpawnActor(blueprint, transform).then(
                            SetAutopilot(FutureActor, True)
                        )
                    )

            for response in self.client.apply_batch_sync(batch):
                if response.error:
                    logging.error(response.error)
                else:
                    vehicles_list.append(response.actor_id)

            # -------------
            # Spawn Walkers
            # -------------
            # some settings
            percentagePedestriansRunning = 0.0  # how many pedestrians will run
            percentagePedestriansCrossing = (
                0.0  # how many pedestrians will walk through the road
            )
            # 1. take all the random locations to spawn
            spawn_points = []
            for i in range(self.walkers):
                spawn_point = carla.Transform()
                loc = world.get_random_location_from_navigation()
                if loc != None:
                    spawn_point.location = loc
                    spawn_points.append(spawn_point)
            # 2. we spawn the walker object
            batch = []
            walker_speed = []
            for spawn_point in spawn_points:
                walker_bp = random.choice(blueprintsWalkers)
                # set as not invencible
                if walker_bp.has_attribute("is_invincible"):
                    walker_bp.set_attribute("is_invincible", "false")
                # set the max speed
                if walker_bp.has_attribute("speed"):
                    if random.random() > percentagePedestriansRunning:
                        # walking
                        walker_speed.append(
                            walker_bp.get_attribute("speed").recommended_values[1]
                        )
                    else:
                        # running
                        walker_speed.append(
                            walker_bp.get_attribute("speed").recommended_values[2]
                        )
                else:
                    print("Walker has no speed")
                    walker_speed.append(0.0)
                batch.append(SpawnActor(walker_bp, spawn_point))
            results = self.client.apply_batch_sync(batch, True)
            walker_speed2 = []
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error)
                else:
                    walkers_list.append({"id": results[i].actor_id})
                    walker_speed2.append(walker_speed[i])
            walker_speed = walker_speed2
            # walker_speed2 = []
            # walker_speed = []
            # 3. we spawn the walker controller
            batch = []
            walker_controller_bp = world.get_blueprint_library().find(
                "controller.ai.walker"
            )
            for i in range(len(walkers_list)):
                batch.append(
                    SpawnActor(
                        walker_controller_bp, carla.Transform(), walkers_list[i]["id"]
                    )
                )
            results = self.client.apply_batch_sync(batch, True)
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error)
                else:
                    walkers_list[i]["con"] = results[i].actor_id
            # 4. we put altogether the walkers and controllers id to get the objects from their id
            for i in range(len(walkers_list)):
                all_id.append(walkers_list[i]["con"])
                all_id.append(walkers_list[i]["id"])
            all_actors = world.get_actors(all_id)
            # all_actors = world.get_actors(all_id)

            # wait for a tick to ensure self.client receives the last transform of the walkers we have just created

            # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
            # set how many pedestrians can cross the road
            world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
            for i in range(0, len(all_id), 2):
                # start walker
                all_actors[i].start()
                # set walk to random point
                all_actors[i].go_to_location(
                    world.get_random_location_from_navigation()
                )
                # max speed
                all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

            print(
                "spawned %d vehicles and %d walkers, press Ctrl+C to exit."
                % (len(vehicles_list), len(walkers_list))
            )

            while self.running:
                world.wait_for_tick(100)

        finally:
            print("\ndestroying %d vehicles" % len(vehicles_list))
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in vehicles_list]
            )

            # stop walker controllers (list is [controller, actor, controller, actor ...])
            for i in range(0, len(all_id), 2):
                all_actors[i].stop()

            print("\ndestroying %d walkers" % len(walkers_list))
            self.client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

            time.sleep(0.5)

    def stop(self) -> None:
        self.running = False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = CarlaClient()
    spawner = spawn(client)
    input("press enter to quit")
