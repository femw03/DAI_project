from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass
from typing import Any, Optional, Type

import carla

from .carla_blueprint import CarlaBlueprint

# Import must be done like this, otherwise the following import error occurs:
# ModuleNotFoundError: No module named 'carla.libcarla.command'
# https://github.com/carla-simulator/carla/issues/6414
SPAWN_ACTOR = carla.command.SpawnActor
FUTURE_ACTOR = carla.command.FutureActor
SET_AUTO_PILOT = carla.command.SetAutopilot
DESTROY_ACTOR = carla.command.DestroyActor


class CarlaCommandResponse:
    def __init__(self, response) -> None:
        self.response = response

    @property
    def error(self) -> Optional[str]:
        return self.response.error

    @property
    def actor_id(self) -> int:
        return self.response.actor_id


class CarlaCommand(ABC):
    def __init__(self, command: Any) -> None:
        super().__init__()
        self.command = command
        self.next: Optional[CarlaCommand] = None

    def then(self, command: CarlaCommand) -> CarlaCommand:
        next_command = self
        while next_command.next is not None:
            next_command = next_command.next

        next_command.next = command
        return self

    @property
    def reduced(self) -> carla.command:
        next_command = self.next
        base = self.command
        while next_command is not None:
            base = base.then(next_command.command)
            next_command = next_command.next
        return base


class SpawnActorCommand(CarlaCommand):
    def __init__(self, blueprint: CarlaBlueprint, location: carla.Transform) -> None:
        command = SPAWN_ACTOR(blueprint.blueprint, location)
        super().__init__(command)


class SetAutoPiloteCommand(CarlaCommand):
    def __init__(self, setActive: bool) -> None:
        super().__init__(SET_AUTO_PILOT(FUTURE_ACTOR, setActive))


class DestroyActorCommand(CarlaCommand):
    def __init__(self, actor: Type["CarlaActor"]) -> None:  # type: ignore  # noqa: F821
        super().__init__(DESTROY_ACTOR(actor.actor))


@dataclass
class CarlaLocation:
    x: float
    y: float
    z: float

    @staticmethod
    def from_native(location: carla.Location) -> CarlaLocation:
        return CarlaLocation(location.x, location.y, location.z)

    @property
    def native(self) -> carla.Location:
        return carla.Location(self.x, self.y, self.z)


class CarlaVector3D:
    """A wrapper around the carla 3D vector object"""

    def __init__(self, vector: carla.Vector3D) -> None:
        assert isinstance(vector, carla.Vector3D)
        self.vector = vector
        self.x = vector.x
        self.y = vector.y
        self.z = vector.z

    @property
    def magnitude(self) -> float:
        """Calculate the magnitude of the vector"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
