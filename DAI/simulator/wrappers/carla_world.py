from __future__ import annotations
from typing import Optional

import carla
import logging

from .carla_actor import CarlaActor
from .carla_blueprint import CarlaBlueprintLibrary

logger = logging.getLogger(__name__)


class CarlaWorld:
    """A wrapper class around the carla world object to allow Typing"""

    def __init__(self, world: carla.World) -> None:
        assert isinstance(
            world, carla.World
        ), f"Instantiate a CarlaWorld object with an actual carla world object, received {type(world)}"
        self.world = world
        self._map: Optional[CarlaMap] = None

    def tick(self, timeout=10) -> int:
        """Signal the server to compute the next simulation tick, returns the ID of the computed frame"""
        logger.debug("Signaling world tick")
        tick_id = self.world.tick(timeout)
        logger.debug(f"Received world tick {tick_id}")
        return tick_id

    def spawn_actor(self, blueprint, location) -> CarlaActor:
        try:
            actor = self.world.spawn_actor(blueprint, location)
        except Exception as e:
            logger.error(
                f"The following exception occurred when spawning {blueprint}: {e}"
            )
        return CarlaActor(actor)

    @property
    def blueprint_library(self) -> CarlaBlueprintLibrary:
        return self.world.get_blueprint_library()

    @property
    def map(self) -> CarlaMap:
        if self._map is None:
            self._map = CarlaMap(self.world.get_map())
        return self._map


class CarlaMap:
    """Wrapper class around carla.Map"""

    def __init__(self, map: carla.Map) -> None:
        assert isinstance(map, carla.Map), f"Expected carla.Map but got {type(map)}"
        self.map = map

    @property
    def spawn_points(self) -> list[carla.Transform]:
        return self.map.get_spawn_points()
