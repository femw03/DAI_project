from __future__ import annotations

from abc import ABC, abstractmethod
from numpy.typing import ArrayLike
from typing import Callable, final, List


class Image(ABC):
    @abstractmethod
    def getImageBytes() -> ArrayLike:
        """Get the data from an image as a numpy array.

        Returns:
            ArrayLike: a array representing the image
        """
        pass
    
class Lidar(ABC):
    @abstractmethod
    def getLidarBytes() -> ArrayLike:
        """
        Get the data from the LIDAR data as an array.
        """
    
    
class CarlaBridge(ABC):
    """
    The bridge between Carla and the python environment.
    When an image is created self.onImageReceived is called
    """
    def __init__(self, onImageReceived: Callable[[Image, Lidar], None]) -> None:
        """Create an instance of CarlaBridge

        Args:
            onImageReceived (Callable[[Image], None]): Called when Carla emits an image
        """
        super().__init__()
        self.onImageReceived = onImageReceived
        
    @abstractmethod
    def _addImage(self, image: Image, lidar: Lidar) -> None:
        """Add an image to the internal buffer of the file

        Args:
            image (Image): an image
        """
        pass  
        
    @final    
    def addImage(self, image: Image, lidar: Lidar) -> None:
        """Adds an image to the internal buffer and calls self.onReceive

        Args:
            image (Image): the image that is added to the buffer
        """
        self._addImage(image, lidar)
        self.onImageReceived(image, lidar)

    
    @abstractmethod
    def setSpeed(speed: float) -> None:
        """set the speed to the given float

        Args:
            speed (float): gas between 0-1
        """
        pass
    
class BoundingBox():
    """
    This is an object that represents a bounding box 
    which is fully defined by the x1,x2,y1,y2 parameters
    """
    def __init__(self, x1: float, x2: float, y1: float, y2: float) -> None:
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        
class Object():
    """This is the result of computer vision module box"""
    def __init__(self, type: str, boundingBox: BoundingBox, confidence: float, distance: float) -> None:
        """Create an object of the incoming image data

        Args:
            type (str): the classification type of this object
            boundingBox (BoundingBox): the bounding box for this object
            confidence (float): confidence that classification is correct
            distance (float): distance to nearest point of this object
        """
        self.type = type
        self.boundingBox = boundingBox
        self.confidence = confidence
        self.distance = distance

class CVBridge(ABC):
    """
    This is the service that consumes images and returns a list of objects
    """
    def __init__(self, onProcessingFinished: Callable[[List[Object], Image, Lidar], None]) -> None:
        """Create an instance of the CVBridge where the onProcessingFinished function is called
        When a picture is finished being processed into objects

        Args:
            onProcessingFinished (Callable[[List[Object]], None]): Called when the images is done processing
        """
        super().__init__()
        self.onProcessingFinished = onProcessingFinished
        
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
        self.onProcessingFinished(objects)