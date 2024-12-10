"""Contains helper functions to enhance visualisations"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import cv2
import numpy as np

from ..interfaces import BoundingBox, Object, ObjectType


@dataclass
class ObjectDTO:
    bounding_box: BoundingBox
    type: ObjectType
    angle: float
    distance: float
    is_relevant: bool
    confidence: float

    @staticmethod
    def from_object(object: Object, is_relevant: bool) -> ObjectDTO:
        return ObjectDTO(
            bounding_box=object.boundingBox,
            angle=object.angle,
            distance=object.distance,
            type=object.type,
            confidence=object.confidence,
            is_relevant=is_relevant,
        )


FONT_SCALE = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX


def add_object_information(image: np.ndarray, objects: List[ObjectDTO]) -> np.ndarray:
    """Creates a new image with object bounding boxes and object information drawn"""
    image = image.copy()

    for object in objects:
        box = object.bounding_box
        color = enum_to_color(object.type)
        cv2.rectangle(
            image,
            (box.x1, box.y1),
            (box.x2, box.y2),
            color=color,
            thickness=5 if object.is_relevant else 1,
        )
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


def add_static_information(image: np.ndarray, data: Dict[str, str]) -> np.ndarray:
    """A the static information of the data in the LHS of the screen"""
    image = image.copy()
    rectangle_height = len(data) * 15 + 30
    cv2.rectangle(
        image,
        (0, 0),
        (200, rectangle_height),
        color=(255, 255, 255),
        thickness=cv2.FILLED,
    )
    y_pos = 15
    for key in data.keys():
        cv2.putText(
            image,
            f"{key} = {data[key]}",
            (10, y_pos),
            FONT,
            FONT_SCALE,
            (0, 0, 0),
            1,
        )
        y_pos += 15
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
