"""Contains helper functions to enhance visualisations"""

from typing import List

import cv2
import numpy as np

from ..interfaces import CarlaFeatures, Object

FONT_SCALE = 0.5


def add_object_information(image: np.ndarray, objects: List[Object]) -> np.ndarray:
    """Creates a new image with object bounding boxes and object information drawn"""
    image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for object in objects:
        box = object.boundingBox
        cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), color=(255, 0, 0))
        cv2.putText(
            image,
            f"{object.type.name}\n{object.angle:.2f}",
            (box.x2, box.y1),
            font,
            FONT_SCALE,
            (255, 0, 0),
            1,
        )
    return image


def add_static_information(image: np.ndarray, data: CarlaFeatures) -> np.ndarray:
    """A the static information of the data in the LHS of the screen"""
    image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(image, (0, 0), (200, 60), color=(255, 255, 255), thickness=cv2.FILLED)
    cv2.putText(
        image,
        f"current speed = {data.current_speed}",
        (10, 15),
        font,
        FONT_SCALE,
        (0, 0, 0),
        1,
    )
    cv2.putText(
        image,
        f"max speed = {data.max_speed}",
        (10, 30),
        font,
        FONT_SCALE,
        (0, 0, 0),
        1,
    )
    cv2.putText(
        image,
        f"light = {data.current_light}",
        (10, 45),
        font,
        FONT_SCALE,
        (0, 0, 0),
        1,
    )
    return image
