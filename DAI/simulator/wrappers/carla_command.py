from __future__ import annotations
from typing import Any, Optional

import carla
import logging

from .carla_blueprint import CarlaBlueprint

from abc import ABC

# Import must be done like this, otherwise the following import error occurs:
# ModuleNotFoundError: No module named 'carla.libcarla.command'
# https://github.com/carla-simulator/carla/issues/6414
SPAWN_ACTOR = carla.command.SpawnActor
FUTURE_ACTOR = carla.command.FutureActor
SET_AUTO_PILOT = carla.command.SetAutopilot

logger = logging.getLogger(__name__)


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

    @property
    def reduces(self) -> carla.command:
        next_command = self
        base = self.command
        while next_command is not None:
            base = base.then(next_command.command)
            next_command = next_command.next
        return base


class SpawnActorCommand(CarlaCommand):
    def __init__(self, blueprint: CarlaBlueprint, location: carla.Transform) -> None:
        command = carla.command.SpawnActor(blueprint.blueprint, location)
        super().__init__(command)


class SetAutoPiloteCommand(CarlaCommand):
    def __init__(self, setActive: bool) -> None:
        super().__init__(
            carla.command.SetAutoPilot(carla.command.FutureActor, setActive)
        )
