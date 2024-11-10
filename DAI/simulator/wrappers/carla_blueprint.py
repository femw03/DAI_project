from __future__ import annotations
from typing import List

import carla
import logging

logger = logging.getLogger(__name__)


class CarlaBlueprintLibrary:
    def __init__(self, blueprint_library=carla.BlueprintLibrary) -> None:
        assert isinstance(
            blueprint_library, carla.BlueprintLibrary
        ), f"Received {type(blueprint_library)} instead of carla.BlueprintLibrary"
        self.blueprint_library = blueprint_library

    def filter(self, search_string) -> List[CarlaBlueprint]:
        blueprints = [
            CarlaBlueprint(blueprint)
            for blueprint in self.blueprint_library.filter(search_string)
        ]
        logger.debug(
            f"The search_string '{search_string}' returned {len(blueprints)} blueprints: {blueprints}"
        )
        return blueprints


class CarlaBlueprint:
    def __init__(self, blueprint: carla.ActorBlueprint) -> None:
        assert isinstance(
            blueprint, carla.ActorBlueprint
        ), f"Received {type(blueprint)} but expected carla.ActorBlueprint"
        self.blueprint = blueprint

    def __getitem__(self, id: str) -> carla.ActorAttribute:
        return self.blueprint.get_attribute(id)

    def __setitem__(self, id: str, value: carla.ActorAttribute) -> None:
        self.blueprint.set_attribute(id, value)

    def contains(self, id: str) -> bool:
        return self.blueprint.has_attribute(id)

    def __repr__(self) -> str:
        return self.blueprint.id
