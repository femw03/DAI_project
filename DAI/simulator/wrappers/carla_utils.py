from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Type

import carla
import numpy as np

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

    def distance_to(self, other: CarlaLocation) -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        )

    def vector_to(self, other: CarlaLocation) -> CarlaVector3D:
        return CarlaVector3D.fromxyz(
            other.x - self.x, other.y - self.y, other.z - self.z
        )

    @property
    def array(self) -> np.ndarray:
        return np.array((self.x, self.y, self.z))

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z))

    def __str__(self) -> str:
        return f"({self.x}, {self.y}, {self.z})"

    def __repr__(self) -> str:
        return str(self)


class CarlaVector3D:
    """A wrapper around the carla 3D vector object"""

    def __init__(self, vector: carla.Vector3D) -> None:
        assert isinstance(vector, carla.Vector3D)
        self.vector = vector
        self.x = vector.x
        self.y = vector.y
        self.z = vector.z
        self.array = np.array((self.x, self.y, self.z))

    @property
    def magnitude(self) -> float:
        """Calculate the magnitude of the vector"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    @staticmethod
    def fromxyz(x: float, y: float, z: float) -> CarlaVector3D:
        return CarlaVector3D(carla.Vector3D(x, y, z))

    def __hash__(self) -> int:
        return hash(self.array)


class CarlaWaypoint:
    """https://carla.readthedocs.io/en/latest/python_api/#carlawaypoint"""

    def __init__(self, waypoint: carla.Waypoint) -> None:
        assert isinstance(waypoint, carla.Waypoint)
        self.waypoint = waypoint

    def get_next(self, distance: float) -> List[CarlaWaypoint]:
        """Get the next waypoints within the distances"""
        return [CarlaWaypoint(wp) for wp in self.waypoint.next(distance)]

    @property
    def transform(self) -> carla.Transform:
        return self.waypoint.transform

    @property
    def road_id(self) -> int:
        return self.waypoint.road_id

    @property
    def section_id(self) -> int:
        return self.waypoint.section_id

    @property
    def junction_id(self) -> int:
        return self.waypoint.junction_id

    @property
    def lane_id(self) -> int:
        return self.waypoint.lane_id

    @property
    def is_junction(self) -> bool:
        return self.waypoint.is_junction

    @property
    def right_lane_marking(self) -> CarlaLaneMarking:
        return CarlaLaneMarking(self.waypoint.right_lane_marking)

    @property
    def left_lane_marking(self) -> CarlaLaneMarking:
        return CarlaLaneMarking(self.waypoint.left_lane_marking)

    @property
    def left_lane(self) -> Optional[CarlaWaypoint]:
        wp = self.waypoint.get_left_lane()
        return CarlaWaypoint(wp) if wp is not None else None

    @property
    def right_lane(self) -> Optional[CarlaWaypoint]:
        wp = self.waypoint.get_right_lane()
        return CarlaWaypoint(wp) if wp is not None else None

    @property
    def lane_type(self) -> CarlaLaneType:
        return CarlaLaneType.from_native(self.waypoint.lane_type)

    @property
    def location(self) -> CarlaLocation:
        return CarlaLocation.from_native(self.transform.location)


class CarlaLaneMarking:
    """https://carla.readthedocs.io/en/latest/python_api/#carla.LaneMarking"""

    def __init__(self, marking: carla.LaneMarking) -> None:
        assert isinstance(marking, carla.LaneMarking)
        self.marking = marking

    @property
    def lane_change(self) -> CarlaLaneChange:
        return CarlaLaneChange.from_native(self.marking.lane_change)


class CarlaLaneChange(Enum):
    """https://carla.readthedocs.io/en/latest/python_api/#carla.LaneChange"""

    NONE = carla.LaneChange.NONE
    RIGHT = carla.LaneChange.Right
    LEFT = carla.LaneChange.Left
    BOTH = carla.LaneChange.Both

    @staticmethod
    def from_native(native: carla.LaneChange):
        return next((state for state in CarlaLaneChange if state.value == native), None)


class CarlaLaneType(Enum):
    """https://carla.readthedocs.io/en/latest/python_api/#carla.LaneType"""

    NONE = carla.LaneType.NONE
    DRIVING = carla.LaneType.Driving
    STOP = carla.LaneType.Stop
    SHOULDER = carla.LaneType.Shoulder
    BIKING = carla.LaneType.Biking
    SIDEWALK = carla.LaneType.Sidewalk
    BORDER = carla.LaneType.Border
    RESTRICTED = carla.LaneType.Restricted
    PARKING = carla.LaneType.Parking
    BIDIRECTIONAL = carla.LaneType.Bidirectional
    MEDIAN = carla.LaneType.Median
    SPECIAL1 = carla.LaneType.Special1
    SPECIAL2 = carla.LaneType.Special2
    SPECIAL3 = carla.LaneType.Special3
    ROADWORKS = carla.LaneType.RoadWorks
    TRAM = carla.LaneType.Tram
    RAIL = carla.LaneType.Rail
    ENTRY = carla.LaneType.Entry
    EXIT = carla.LaneType.Exit
    OFFRAMP = carla.LaneType.OffRamp
    ONRAMP = carla.LaneType.OnRamp
    ANY = carla.LaneType.Any

    @staticmethod
    def from_native(native: carla.LaneType):
        return next((state for state in CarlaLaneType if state.value == native), None)
