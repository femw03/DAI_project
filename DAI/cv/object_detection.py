"""This file contains functions for object detection using yolo"""

from typing import Dict, List, Tuple

import torch
from ultralytics import YOLO

from ..interfaces import BoundingBox, CarlaData, Object, ObjectType
from .calculate_distance import (
    calculate_anlge,
    calculate_object_distance,
)
from .stabilizer import DetectionStabilizer


def big_object_detection(
    model: YOLO, data: CarlaData, tracker: DetectionStabilizer
) -> List[Object]:
    """Do object detection for the general big net"""
    results = model.predict(
        data.rgb_image.get_image_bytes(),
        half=True,
        device="cuda:0",
        verbose=False,
        conf=0.3,
        iou=0.8,
    )

    # Convert the YOLO Results to a list of Objects
    result = results[0]
    probabilities: List[float] = result.boxes.conf.tolist()
    label_indices: List[int] = result.boxes.cls.tolist()
    labels: Dict[int, str] = result.names
    bounding_box_coords: List[List[float]] = result.boxes.xyxy.to(torch.int).tolist()

    detections: List[Tuple[List[float], float, str]] = []

    for confidence, label_index, bounding_box_coord in zip(
        probabilities, label_indices, bounding_box_coords
    ):
        type = ObjectType.label(labels[label_index])
        detections.append(
            (
                type,
                confidence,
                BoundingBox.from_array(bounding_box_coord),
            )
        )
    tracks = tracker.stabilize_detections(detections)
    the_real_ones = []
    for type, confidence, box in tracks:
        # print(track.get_det_class())
        assert type is not None
        assert confidence is not None
        object_location = calculate_object_distance(data.depth_data, box)
        # object_location = ObjectDistance(depth=0, location=(0, 0))
        object_angle = calculate_anlge(
            object_location.location[0],
            data.rgb_image.fov,
            data.depth_data.get_depth_bytes().shape[1],
        )
        the_real_ones.append(
            Object(
                type=type,
                confidence=confidence,
                angle=object_angle,
                distance=object_location.depth,
                boundingBox=box,
            )
        )
    return the_real_ones
