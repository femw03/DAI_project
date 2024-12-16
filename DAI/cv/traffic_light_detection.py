"""This file handles object detection specific to traffic lights and their relevance"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

import torch
from loguru import logger
from ultralytics import YOLO

from ..interfaces import BoundingBox, CarlaData, Object, ObjectType
from .calculate_distance import calculate_anlge, calculate_object_distance
from .stabilizer import DetectionStabilizer


class TrafficLight(Enum):
    GREEN_RELEVANT = ("GREEN", True, "traffic light green relevant")
    GREEN_IRRELEVANT = ("GREEN", False, "traffic light green not relevant")
    YELLOW_RELEVANT = ("YELLOW", True, "traffic light yellow relevant")
    YELLOW_IRRELEVANT = ("YELLOW", False, "traffic light yellow not relevant")
    RED_RELEVANT = ("RED", True, "traffic light red relevant")
    RED_IRRELEVANT = ("RED", False, "traffic light red not relevant")

    def __init__(
        self, color: Literal["RED", "GREEN", "YELLOW"], is_relevant: bool, label
    ) -> None:
        self.color = color
        self.is_relevant = is_relevant
        self.label = label

    @staticmethod
    def from_label(label: str) -> Optional[None]:
        for light in TrafficLight:
            if light.label == label:
                return light

        logger.warning(f"Tried to instantiate Traffic light with label {label}")
        return None

    @staticmethod
    def should_stop(traffic_lights: List[TrafficLight]) -> bool:
        """Determine if the traffic_lights list indicates if the car should stop"""
        relevant_traffic_lights = [
            traffic_light
            for traffic_light in traffic_lights
            if traffic_light.is_relevant
        ]
        should_stop = any(
            [
                traffic_light.color == "RED" or "YELLOW"
                for traffic_light in relevant_traffic_lights
            ]
        )
        return should_stop


def detect_traffic_lights(
    model: YOLO, stabilizer: DetectionStabilizer, data: CarlaData
) -> Tuple[List[TrafficLight], List[Object]]:
    detection_result = model.predict(
        data.rgb_image.get_image_bytes(),
        half=True,
        device="cuda:0",
        verbose=False,
        conf=0.3,
    )
    result = detection_result[0]
    probabilities: List[float] = result.boxes.conf.tolist()
    label_indices: List[int] = result.boxes.cls.tolist()
    labels: Dict[int, str] = result.names
    bounding_box_coords: List[List[float]] = result.boxes.xyxy.to(torch.int).tolist()

    detections: List[Tuple[List[float], float, str]] = []

    for confidence, label_index, bounding_box_coord in zip(
        probabilities, label_indices, bounding_box_coords
    ):
        type = labels[label_index]
        detections.append(
            (
                type,
                confidence,
                BoundingBox.from_array(bounding_box_coord),
            )
        )
    tracks = stabilizer.stabilize_detections(detections)
    the_real_ones = []
    traffic_lights = []
    for type, confidence, box in tracks:
        object_location = calculate_object_distance(data.lidar_data, box)
        # object_location = ObjectDistance(depth=0, location=(0, 0))
        object_angle = calculate_anlge(
            object_location.location[0],
            data.rgb_image.fov,
            data.lidar_data.get_lidar_bytes().shape[1],
        )
        the_real_ones.append(
            Object(
                type=ObjectType.TRAFFIC_LIGHT,
                confidence=confidence,
                angle=object_angle,
                distance=object_location.depth,
                boundingBox=box,
            )
        )
        traffic_lights.append(TrafficLight.from_label(type))

    traffic_lights = [
        TrafficLight.from_label(labels[label_index]) for label_index in label_indices
    ]

    return traffic_lights, the_real_ones
