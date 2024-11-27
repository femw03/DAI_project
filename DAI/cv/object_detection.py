"""This file contains functions for object detection using yolo"""

from typing import Dict, List

import torch
from ultralytics import YOLO

from ..interfaces import BoundingBox, CarlaData, Object, ObjectType
from .calculate_distance import calculate_anlge, calculate_object_distance


def big_object_detection(model: YOLO, data: CarlaData) -> List[Object]:
    """Do object detection for the general big net"""
    results = model.predict(
        data.rgb_image.get_image_bytes(), half=True, device="cuda:0", verbose=False
    )

    # Convert the YOLO Results to a list of Objects
    detected: List[Object] = []
    result = results[0]
    probabilities: List[float] = result.boxes.conf.tolist()
    label_indices: List[int] = result.boxes.cls.tolist()
    labels: Dict[int, str] = result.names
    bounding_box_coords: List[List[float]] = result.boxes.xyxy.to(torch.int).tolist()

    for confidence, label_index, bounding_box_coord in zip(
        probabilities, label_indices, bounding_box_coords
    ):
        type = ObjectType.label(labels[label_index])
        bounding_box = BoundingBox.from_array(bounding_box_coord)
        object_location = calculate_object_distance(
            data.lidar_data.get_lidar_bytes(), bounding_box
        )
        object_angle = calculate_anlge(
            object_location.location[0],
            data.rgb_image.fov,
            data.lidar_data.get_lidar_bytes().shape[1],
        )
        detected.append(
            Object(
                type=type,
                confidence=confidence,
                angle=object_angle,
                distance=object_location.depth,
                boundingBox=bounding_box,
            )
        )
    return detected
