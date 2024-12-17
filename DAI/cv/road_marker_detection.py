"""File that detects markers from the road such as stopping lines and pedestrian crossings"""

from __future__ import annotations

from typing import Dict, List

import torch
from ultralytics import YOLO

from ..interfaces import BoundingBox, CarlaData, Object, ObjectType
from .calculate_distance import calculate_anlge, calculate_object_distance

LABELS = {
    "zebra_crossing": ObjectType.CROSSING,
    "stop_line": ObjectType.STOP_LINE,
}


def detect_road_markers(model: YOLO, data: CarlaData) -> List[Object]:
    """Additionally detect stop line and crossing markers with a seperate model"""
    results = model.predict(
        data.rgb_image.get_image_bytes(),
        half=True,
        device="cuda:0",
        verbose=False,
        conf=0.5,
        # iou=0.3,
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
        type = LABELS[labels[label_index]]
        bounding_box = BoundingBox.from_array(bounding_box_coord)
        object_location = calculate_object_distance(data.depth_data, bounding_box)
        # object_location = ObjectDistance(depth=0, location=(0, 0))
        object_angle = calculate_anlge(
            object_location.location[0],
            data.rgb_image.fov,
            data.depth_data.get_depth_bytes().shape[1],
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
