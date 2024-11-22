from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, final

import numpy as np
from loguru import logger


class Image(ABC):
    def __init__(self, width: int, height: int, fov: int) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.fov = fov

    @abstractmethod
    def get_image_bytes(self) -> np.ndarray:
        """Get the data from an image as a numpy array. HWC format BGR [0, 255], (Height, Width, # Channels)

        Returns:
            np.ndarray: a array representing the image
        """
        pass


class Lidar(ABC):
    def __init__(self, width: int, height: int, fov: int) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.fov = fov

    @abstractmethod
    def get_lidar_bytes(self) -> np.ndarray:
        """
        Get the data from the LIDAR data as an array. (Height, Width, # Channels)
        """


@dataclass
class CarlaData:
    """A carrier object that represents the output of the [CarlaBridge]"""

    rgb_image: Image
    lidar_data: Lidar
    current_speed: float


class CarlaWorld(ABC):
    """
    The bridge between Carla and the python environment.
    The latest frame data will be present in self.data or can be listened to in by adding a callback
    via self.add_listeners.
    """

    def __init__(self) -> None:
        """Create an instance of CarlaBridge"""
        super().__init__()
        self.__data_listeners: List[Callable[[CarlaData], None]] = []
        self.__tick_listeners: List[Callable[[], None]] = []
        self.data: CarlaData | None = None
        self._speed: float = 0.5

    @final
    def _set_data(self, data: CarlaData) -> None:
        """Sets self.data and notifies listeners

        Args:
            image (Image): the image that is added to the buffer
        """
        self.data = data
        for listener in self.__data_listeners:
            listener(data)

    @final
    def set_speed(self, speed: float) -> None:
        """Sets the ego vehicle speed"""
        self._speed = speed

    @final
    def add_listener(self, callback: Callable[[CarlaData], None]) -> None:
        self.__data_listeners.append(callback)

    @final
    def add_tick_listener(self, callback: Callable[[], None]) -> None:
        self.__tick_listeners.append(callback)

    @final
    def remove_tick_listener(self, callback: Callable[[], None]) -> None:
        try:
            self.__tick_listeners.remove(callback)
        except ValueError:
            logger.warning("Tried to remove a listener that was not present")

    @final
    def await_next_tick(self) -> None:
        """Blocks thread until the next tick has happened"""
        has_ticked = False

        def set_has_ticked():
            global has_ticked
            has_ticked = True

        self.add_tick_listener(set_has_ticked)
        while not has_ticked:
            time.sleep(0)
        self.remove_tick_listener(set_has_ticked)

    @final
    def _notify_tick_listeners(self) -> None:
        """Notify all tick listeners, MUST be called after every world tick"""
        for listener in self.__tick_listeners:
            listener()


@dataclass
class BoundingBox:
    """
    This is an object that represents a bounding box
    which is fully defined by the x1,x2,y1,y2 parameters
    """

    x1: float
    x2: float
    y1: float
    y2: float

    @staticmethod
    def from_array(array: np.ndarray):
        """Create a bounding box from a xyxy array"""
        assert len(array) == 4
        return BoundingBox(array[0], array[2], array[1], array[3])


class ObjectType(str, Enum):
    VEHICLE = "vehicle"
    TRAFFIC_LIGHT = "traffic light"
    TRAFFIC_SIGN = "traffic sign"
    PEDESTRIAN = "pedestrian"


@dataclass
class Object:
    """This is the result of computer vision module box"""

    type: "ObjectType"
    """The classification type of this object"""

    boundingBox: "BoundingBox"
    """The bounding box for this object"""

    confidence: float
    """Confidence that the classification is correct"""

    distance: float
    """Distance to the nearest point of this object"""

    angle: float
    """Lateral angle to the object in radians"""


@dataclass
class CarlaFeatures:
    """
    This represents the set of features that needs to be extracted from [CarlaData]
    This features are the input for the RL-agent
    """

    objects: List[Object]
    current_speed: float
    max_speed: Optional[float]


class ComputerVisionModule(ABC):
    """
    This is the service that contains a method to convert CarlaData into CarlaFeatures.
    """

    @abstractmethod
    def process_data(self, data: CarlaData) -> CarlaFeatures:
        """From the data from the Carla Simulator return a set of features"""
        pass


class CruiseControlAgent(ABC):
    """A CruiseControlAgent converts CarlaFeatures into a speed"""

    @abstractmethod
    def get_action(self, state: CarlaFeatures) -> float:
        """Calculate the throttle speed (0 - 1) from the given state"""
