"""Contains a bunch of dataclass objects that are used to pass information between the parts of the system"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

import numpy as np
import torch
from loguru import logger

from .image import Image, Lidar


### Objects ###
class ObjectType(Enum):
    CAR = ["car", "other vehicle", "vehicles"]
    MOTOR_CYCLE = ["motorcycle"]
    BICYLE = ["bicycle"]
    BUS = ["bus", "large vehicles"]
    PEDESTRIAN = ["pedestrians", "other person"]
    RIDER = ["rider"]
    TRAFFIC_LIGHT = ["traffic lights"]
    TRAFFIC_SIGN = ["traffic signs"]
    TRAIN = ["train"]
    TRUCK = ["truck"]
    TRAILER = ["trailer"]

    def __init__(self, labels: List[str]):
        self._labels = labels

    @staticmethod
    def label(label: str) -> Optional[ObjectType]:
        for type in ObjectType:
            if label in type._labels:
                return type
        logger.warning(
            f"Tried to find the type for the label: {label} but it was not found"
        )
        return None


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


@dataclass
class Object:
    """This is the result of computer vision module box"""

    type: ObjectType
    """The classification type of this object"""

    boundingBox: BoundingBox
    """The bounding box for this object"""

    confidence: float
    """Confidence that the classification is correct"""

    distance: float
    """Distance to the nearest point of this object"""

    angle: float
    """Lateral angle to the object in radians"""


### Carla Data ###
@dataclass
class CarlaData:
    """A carrier object that represents the output of the [CarlaWorld]"""

    rgb_image: Image
    lidar_data: Lidar
    current_speed: float
    time_stamp: datetime


### Carla Features ###
@dataclass
class CarlaObservation:
    """
    This represents the set of features that needs to be extracted from [CarlaData]
    This features are the input for the RL-agent
    """

    objects: List[Object]
    current_speed: float
    angle: float
    max_speed: Optional[float]
    stop_flag: bool
    distance_to_stop: Optional[float]
    pedestrian_crossing_flag: bool
    distance_to_pedestrian_crossing: Optional[float]


class AgentFeatures(ABC):
    """An object that carries the features, only requires the creation of tensors"""

    @abstractmethod
    def to_tensor(self) -> torch.FloatTensor:
        """Converts the features into a tensor that represents the input of our model"""
        pass
