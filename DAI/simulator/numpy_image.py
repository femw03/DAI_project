from typing import Callable

import numpy as np

from ..interfaces import Image, Lidar


class NumpyImage(Image):
    """Save an image as a (width, height, 3) image"""

    def __init__(self, data: np.ndarray, fov: int) -> None:
        super().__init__(width=data.shape[1], height=data.shape[0], fov=fov)
        self.data = data

    def get_image_bytes(
        self,
    ) -> np.ndarray:
        return self.data


class NumpyLidar(Lidar):
    """Save an Lidar as a (width, height, 3) image"""

    def __init__(
        self, data: np.ndarray, fov: int, converter: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        super().__init__(width=data.shape[1], height=data.shape[0], fov=fov)
        self.data = data
        self.converter = converter
        """A function that encodes the conversion from 0-1 to meters"""

    def get_lidar_bytes(
        self,
    ) -> np.ndarray:
        return self.data

    def get_lidar_meters(self) -> np.ndarray:
        return self.converter(self.data)
