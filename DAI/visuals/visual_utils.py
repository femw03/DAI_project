from typing import List

import cv2
import numpy as np

from ..interfaces import Object


def add_object_information(image: np.ndarray, objects: List[Object]) -> np.ndarray:
    """Creates a new image with object bounding boxes and object information drawn"""
    image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for object in objects:
        box = object.boundingBox
        cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), color=(255, 0, 0))
        cv2.putText(
            image,
            f"{object.type}\n{object.angle}",
            (box.x2, box.y1),
            font,
            0.25,
            (255, 0, 0),
            1,
        )
    return image
