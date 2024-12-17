"""Abstract implementation agnostic definition for images and depth information"""

from abc import ABC, abstractmethod
from typing import Tuple

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

    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        pass


class Depth(ABC):
    """Abstract carrier for depth information, get_depth_bytes method returns the data as a numpy array of depth meter data"""

    def __init__(self, width: int, height: int, fov: int) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.fov = fov

    @abstractmethod
    def get_depth_bytes(self) -> np.ndarray:
        """
        Get the data from the Depth data as an array. (Height, Width) the unit values should be bounded between 0-1
        """
        pass

    @abstractmethod
    def get_depth_meters(self) -> np.ndarray:
        """
        Get the data from the Depth data as an (Height, Width) array with the values representing meters
        """
