"""Contains helper functions to enhance visualisations"""

from __future__ import annotations

import hashlib
import math as m
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import cv2
import numpy as np

from ..cv import expected_deviation
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
        (400, rectangle_height),
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

# Define the predefined colors for each ObjectType
PREDEFINED_COLORS = {
    ObjectType.CAR: (0, 255, 0),          # Green
    ObjectType.MOTOR_CYCLE: (255, 0, 0),  # Blue
    ObjectType.BICYLE: (255, 255, 0),     # Cyan
    ObjectType.BUS: (0, 255, 0),          # Green
    ObjectType.PEDESTRIAN: (255, 0, 255), # Magenta
    ObjectType.RIDER: (255, 0, 255),      # Magenta
    ObjectType.TRAFFIC_LIGHT: (128, 0, 128), # Purple
    ObjectType.TRAFFIC_SIGN: (0, 165, 255), # Orange
    ObjectType.TRAIN: (255, 255, 255),    # White
    ObjectType.TRUCK: (0, 255, 0),        # Green
    ObjectType.TRAILER: (0, 255, 0),      # Green
    ObjectType.STOP_LINE: (0, 0, 255),    # Red
    ObjectType.CROSSING: (0, 0, 255)      # Red
}

def enum_to_color(enum_value: ObjectType) -> Tuple[int, int, int]:
    """
    Converts an enum value to a predefined color for OpenCV.
    The same enum value will always return the same predefined color.

    Args:
        enum_value: The enum value to be converted to a color.

    Returns:
        A tuple (B, G, R) representing a color in OpenCV format.
    """
    return PREDEFINED_COLORS[enum_value]


def draw_trajectory_line(
    image: np.ndarray,
    depth_information: np.ndarray,
    angle,
    correction_factor,
    boost_factor,
    margin=600,
    at_y=0.3,
    horizon=0.5,
) -> np.ndarray:
    image = image.copy()
    y_count, x_count = depth_information.shape
    sample_horizon_y_coord = int(y_count - y_count * at_y)
    horizon_y_coord = int(y_count * horizon)
    horizon_data = depth_information[sample_horizon_y_coord, :]
    mean, std = horizon_data.mean(), horizon_data.std()
    filtered = horizon_data[np.abs(horizon_data - mean) <= 1 * std]
    distance_to_horizon = filtered.max() * 1000 * 0 + 41.0

    points_to_sample = np.arange(0, y_count - sample_horizon_y_coord)

    def sampler(x):
        assymptote = y_count - horizon_y_coord
        measure_point = y_count - sample_horizon_y_coord
        return distance_to_horizon * (assymptote - measure_point) / (assymptote - x)

    distances = np.array([sampler(point) for point in points_to_sample])

    deviations = np.array(
        [
            expected_deviation(point, angle, correction_factor, boost_factor)
            for point in distances
        ],
    )
    # focal_length = 0.08 / (2 * np.x(np.radians(FOV / 2)))
    # print(distances[0], distances[len(distances) // 2], distances[-1], warp_factor)
    for i, data in enumerate(zip(deviations, distances)):
        deviation, distance = data
        if m.isnan(deviation):
            continue
        x_coord = x_count // 2 + int(deviation)
        # apparent_size = warp_factor / (warp_factor + distance)
        h_margin = int(margin / (distance + 2))
        x_min_coord = x_coord - h_margin
        x_max_coord = x_coord + h_margin
        if x_coord >= 0 and x_coord < x_count - 1:
            y_coord = y_count - i - 1
            image[y_coord, x_coord : min(x_coord + 3, x_count - 1)] = (255, 255, 0)

        if x_max_coord >= 0 and x_max_coord < x_count:
            y_coord = y_count - i - 1
            image[y_coord, x_max_coord : min(x_max_coord + 3, x_count - 1)] = (
                255,
                255,
                0,
            )

        if x_min_coord >= 0 and x_min_coord < x_count:
            y_coord = y_count - i - 1
            image[y_coord, x_min_coord : min(x_min_coord + 3, x_count - 1)] = (
                255,
                255,
                0,
            )
    image[sample_horizon_y_coord, :] = (255, 255, 0)
    image[horizon_y_coord, :] = (0, 255, 0)

    return image
