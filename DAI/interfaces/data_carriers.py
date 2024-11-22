"""Contains a bunch of dataclass objects that are used to pass information between the parts of the system"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

from .image import Image, Lidar


### Objects ###
class ObjectType(str, Enum):
    VEHICLE = "vehicle"
    TRAFFIC_LIGHT = "traffic light"
    TRAFFIC_SIGN = "traffic sign"
    PEDESTRIAN = "pedestrian"


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


### Carla Data ###
@dataclass
class CarlaData:
    """A carrier object that represents the output of the [CarlaWorld]"""

    rgb_image: Image
    lidar_data: Lidar
    current_speed: float


### Carla Features ###
@dataclass
class CarlaFeatures:
    """
    This represents the set of features that needs to be extracted from [CarlaData]
    This features are the input for the RL-agent
    """

    objects: List[Object]
    current_speed: float
    max_speed: Optional[float]
