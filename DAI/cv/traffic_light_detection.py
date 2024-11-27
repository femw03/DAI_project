"""This file handles object detection specific to traffic lights and their relevance"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from loguru import logger
from ultralytics import YOLO

from ..interfaces import Image


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


def detect_traffic_lights(model: YOLO, image: Image) -> List[TrafficLight]:
    detection_result = model.predict(
        image.get_image_bytes(), half=True, device="cuda:0", verbose=False
    )[0]
    label_indices: List[int] = detection_result.boxes.cls.tolist()
    labels: Dict[int, str] = detection_result.names

    traffic_lights = [
        TrafficLight.from_label(labels[label_index]) for label_index in label_indices
    ]

    return traffic_lights
