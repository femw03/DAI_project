"""Contains helper functions to enhance visualisations"""

import hashlib
import random
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..interfaces import CarlaFeatures, Object

FONT_SCALE = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX


def add_object_information(image: np.ndarray, objects: List[Object]) -> np.ndarray:
    """Creates a new image with object bounding boxes and object information drawn"""
    image = image.copy()

    for object in objects:
        box = object.boundingBox
        color = enum_to_color(object.type)
        cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), color=color)
        cv2.putText(
            image,
            f"{object.type.name}:{object.confidence:.0%}",
            (box.x2, box.y1),
            FONT,
            FONT_SCALE,
            color,
            1,
        )
        cv2.putText(
            image,
            f"{object.distance:.1f}m; {object.angle:.2f} \u03c0",
            (box.x2, box.y1 + 15),
            FONT,
            FONT_SCALE,
            color,
            1,
        )
    return image


def add_static_information(
    image: np.ndarray, data: CarlaFeatures, time_per_frame: Optional[float] = None
) -> np.ndarray:
    """A the static information of the data in the LHS of the screen"""
    image = image.copy()
    cv2.rectangle(image, (0, 0), (200, 90), color=(255, 255, 255), thickness=cv2.FILLED)
    cv2.putText(
        image,
        f"current speed = {data.current_speed:.2f}",
        (10, 15),
        FONT,
        FONT_SCALE,
        (0, 0, 0),
        1,
    )
    cv2.putText(
        image,
        f"max speed = {data.max_speed}",
        (10, 30),
        FONT,
        FONT_SCALE,
        (0, 0, 0),
        1,
    )
    cv2.putText(
        image,
        f"light = {data.current_light}",
        (10, 45),
        FONT,
        FONT_SCALE,
        (0, 0, 0),
        1,
    )
    if time_per_frame is not None:
        cv2.putText(
            image,
            f"time = {time_per_frame:.2f}s",
            (10, 60),
            FONT,
            FONT_SCALE,
            (0, 0, 0),
            1,
        )
    return image


def enum_to_color(enum_value: Enum) -> Tuple[int, int, int]:
    """
    Converts an enum value to a consistent random color for OpenCV.
    The same enum value will always return the same color.

    Args:
        enum_value: The enum value to be converted to a color.

    Returns:
        A tuple (B, G, R) representing a color in OpenCV format.
    """
    # Hash the enum value to ensure consistency
    hash_object = hashlib.md5(str(enum_value).encode())
    hash_digest = hash_object.hexdigest()

    # Use the hash to generate consistent random values for B, G, and R
    random.seed(int(hash_digest, 16))  # Seed with the hash value
    blue = random.randint(0, 255)
    green = random.randint(0, 255)
    red = random.randint(0, 255)

    return (blue, green, red)
