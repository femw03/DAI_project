"""Abstract implementation agnostic definition for images and depth information"""

from abc import ABC, abstractmethod

import numpy as np


class Image(ABC):
    """Abstract carrier for image information, the get_image_bytes method returns the data as a numpy array"""

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
    """Abstract carrier for depth information, get_lidar_bytes method returns the data as a numpy array of depth meter data"""

    def __init__(self, width: int, height: int, fov: int) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.fov = fov

    @abstractmethod
    def get_lidar_bytes(self) -> np.ndarray:
        """
        Get the data from the LIDAR data as an array. (Height, Width) the unit is in meters
        """
        pass
