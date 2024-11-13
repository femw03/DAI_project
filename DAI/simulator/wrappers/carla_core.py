from __future__ import annotations
from typing import Optional

import carla
import logging

from .carla_blueprint import CarlaBlueprintLibrary, CarlaBlueprint
from .carla_utils import CarlaLocation

logger = logging.getLogger(__name__)


class CarlaWorld:
    """A wrapper class around the carla world object to allow Typing"""

    def __init__(self, world: carla.World) -> None:
        assert isinstance(
            world, carla.World
        ), f"Instantiate a CarlaWorld object with an actual carla world object, received {type(world)}"
        self.world = world
        self._map: Optional[CarlaMap] = None
        self._pededstrian_crossing_factor: Optional[float] = None

    def tick(self, timeout=10) -> int:
        """Signal the server to compute the next simulation tick, returns the ID of the computed frame"""
        logger.debug("Signaling world tick")
        tick_id = self.world.tick(timeout)
        logger.debug(f"Received world tick {tick_id}")
        return tick_id

    def spawn_actor(
        self,
        blueprint: CarlaBlueprint,
        location: carla.Transform,
        parent: Optional[CarlaActor] = None,
    ) -> CarlaActor:
        try:
            actor = self.world.spawn_actor(
                blueprint.blueprint, location, attach_to=parent.actor
            )
        except Exception as e:
            logger.error(
                f"The following exception occurred when spawning {blueprint}: {e}"
            )
            raise e
        return CarlaActor(actor, self)

    def spawn_vehicle(
        self, blueprint: CarlaBlueprint, location: carla.Transform
    ) -> CarlaVehicle:
        try:
            vehicle = self.world.spawn_actor(blueprint.blueprint, location)
        except Exception as e:
            logger.error(
                f"The following exception occurred when spawning {blueprint}: {e}"
            )
            raise e
        return CarlaVehicle(vehicle, self)

    def spawn_walker(
        self, blueprint: CarlaBlueprint, location: carla.Transform
    ) -> CarlaWalker:
        try:
            if self._pededstrian_crossing_factor is None:
                logger.warning(
                    f"The pedestrian crossing factor was not yet set while attempting to spawn a pedestrian"
                )
            vehicle = self.world.spawn_actor(blueprint.blueprint, location)
        except Exception as e:
            logger.error(
                f"The following exception occurred when spawning {blueprint}: {e}"
            )
            raise e
        return CarlaWalker(vehicle, self)

    def get_random_walker_location(self) -> CarlaLocation:
        return CarlaLocation.from_native(
            self.world.get_random_location_from_navigation()
        )

    @property
    def blueprint_library(self) -> CarlaBlueprintLibrary:
        return CarlaBlueprintLibrary(self.world.get_blueprint_library())

    @property
    def map(self) -> CarlaMap:
        if self._map is None:
            self._map = CarlaMap(self.world.get_map())
        return self._map

    @property
    def pedestrian_crossing_factor(self) -> float:
        return self._pededstrian_crossing_factor

    @pedestrian_crossing_factor.setter
    def pedestrian_crossing_factor(self, value: float) -> None:
        self.world.set_pedestrians_cross_factor(value)
        self._pededstrian_crossing_factor = value


class CarlaMap:
    """Wrapper class around carla.Map"""

    def __init__(self, map: carla.Map) -> None:
        assert isinstance(map, carla.Map), f"Expected carla.Map but got {type(map)}"
        self.map = map

    @property
    def spawn_points(self) -> list[carla.Transform]:
        return self.map.get_spawn_points()


class CarlaActor:
    """Wrapper class around a carla actor"""

    def __init__(self, actor: carla.Actor, world: CarlaWorld) -> None:
        assert isinstance(
            actor, carla.Actor
        ), f"Instantiate a CarlaActor with a carla.Actor object instead of {type(actor)}"
        self.actor = actor
        self.world = world

    def __repr__(self) -> str:
        return str(self.actor)

    def destroy(self) -> bool:
        try:
            return self.actor.destroy()
        except Exception as e:
            logger.error(f"Error when dispawning {self.actor}: {e}")
            return False


class CarlaWalkerAI(CarlaActor):
    """An actor that controls a walker"""

    def __init__(
        self, controller: carla.Actor, world: CarlaWorld, walker: CarlaWalker
    ) -> None:
        super().__init__(controller, world)
        assert isinstance(
            controller, carla.WalkerAIController
        ), f"Instantiate CarlaWalker with a walker instead of {type(controller)}"
        self.walker = walker

    def stop(self) -> None:
        self.actor.stop()

    def start(self) -> None:
        self.actor.start()


class CarlaWalker(CarlaActor):
    """"""

    def __init__(self, walker: carla.Walker, world: CarlaWorld) -> None:
        super().__init__(walker, world)
        assert isinstance(
            walker, carla.Walker
        ), f"Instantiate CarlaWalker with a walker instead of {type(walker)}"
        self.controller: Optional[CarlaWalkerAI] = None

    def add_controller(self, blueprint: CarlaBlueprint) -> CarlaWalkerAI:
        controller_actor = self.world.spawn_actor(
            blueprint, carla.Transform(), parent=self
        )
        self.controller = CarlaWalkerAI(controller_actor.actor, self.world, self)
        return self.controller

    def destroy(self) -> bool:
        if self.controller is not None:
            self.controller.stop()
            self.controller.destroy()

        return super().destroy()


class CarlaVehicle(CarlaActor):
    """"""

    def __init__(self, vehicle: carla.Vehicle, world: CarlaWorld) -> None:
        super().__init__(vehicle, world)
        assert isinstance(
            vehicle, carla.Vehicle
        ), f"Instantiate CarlaVehicle with a vehicle instead of {type(vehicle)}"
        self._autopilot = False

    @property
    def autopilot(self) -> bool:
        return self._autopilot

    @autopilot.setter
    def autopilot(self, value: bool) -> None:
        self.actor.set_autopilot(value)
        self._autopilot = value
