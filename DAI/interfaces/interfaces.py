from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, final

import numpy as np


class Image(ABC):
    @abstractmethod
    def get_image_bytes(self) -> np.ndarray:
        """Get the data from an image as a numpy array. HWC format BGR [0, 255], (Height, Width, # Channels)

        Returns:
            np.ndarray: a array representing the image
        """
        pass


class Lidar(ABC):
    @abstractmethod
    def get_lidar_bytes(self) -> np.ndarray:
        """
        Get the data from the LIDAR data as an array. (Height, Width, # Channels)
        """


Speed = float
DataReceivedCallBack = Callable[[Image, Lidar, Speed], None]


class CarlaBridge(ABC):
    """
    The bridge between Carla and the python environment.
    When an image is created self.onImageReceived is called
    """

    def __init__(self, onImageReceived: DataReceivedCallBack) -> None:
        """Create an instance of CarlaBridge

        Args:
            onImageReceived (Callable[[Image], None]): Called when Carla emits an image
        """
        super().__init__()
        self.onImageReceived = onImageReceived

    @abstractmethod
    def _add_image(self, image: Image, lidar: Lidar) -> None:
        """Add an image to the internal buffer of the file

        Args:
            image (Image): an image
        """
        pass

    @final
    def add_image(self, image: Image, lidar: Lidar) -> None:
        """Adds an image to the internal buffer and calls self.onReceive

        Args:
            image (Image): the image that is added to the buffer
        """
        self._add_image(image, lidar)
        self.onImageReceived(image, lidar)

    @abstractmethod
    def set_speed(speed: float) -> None:
        """set the speed to the given float

        Args:
            speed (float): gas between 0-1
        """
        pass


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


ProcessingFinishedCallBack = Callable[[Image, Lidar, List[Object], Speed, Speed], None]
"""A callback that indicates the Computer Vision process finishing, returns the Image, Lidar, Detected Objects, Current Speed and Speed Limit"""


class CVBridge(ABC):
    """
    This is the service that consumes images and returns a list of objects
    """

    def __init__(self, onProcessingFinished: ProcessingFinishedCallBack) -> None:
        """Create an instance of the CVBridge where the onProcessingFinished function is called
        When a picture is finished being processed into objects

        Args:
            onProcessingFinished (Callable[[List[Object]], None]): Called when the images is done processing
        """
        super().__init__()
        self.onProcessingFinished = onProcessingFinished

    @abstractmethod
    def on_data_received(self, image: Image, lidar: Lidar) -> None:
        pass

    @abstractmethod
    def _submitObjects(self, objects: List[Object], image: Image) -> None:
        """Store the object in local memory

        Args:
            object (List[Object]): objects to be stored
            image (Image) : image from where the objects were created
        """
        pass

    @final
    def submitObjects(self, objects: List[Object], image: Image) -> None:
        """Submits the list of objects to be processed by the next steps.
        Calls the self.onProcessingFinished function

        Args:
            objects (List[Object]): objects that finished processing
            image (Image): the image where the objects were created from
        """
        self._submitObjects(objects, image)
        self.onProcessingFinished(objects, image)
