import numpy as np

from ..interfaces import Image, Lidar


class NumpyImage(Image):
    """Save an image as a (width, height, 3) image"""

    def __init__(self, data: np.ndarray) -> None:
        super().__init__()
        self.data = data

    def get_image_bytes(
        self,
    ) -> np.ndarray:
        return self.data


class NumpyLidar(Lidar):
    """Save an Lidar as a (width, height, 3) image"""

    def __init__(self, data: np.ndarray) -> None:
        super().__init__()
        self.data = data

    def get_lidar_bytes(
        self,
    ) -> np.ndarray:
        return self.data
