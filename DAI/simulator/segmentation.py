"""
Contains helper functions to extract bounding boxes and labels from
from segmentation images
"""

from typing import List, Tuple

import cv2
import numpy as np

from ..interfaces import BoundingBox, Object, ObjectType
from ..visuals.visual_utils import add_object_information

mappings = [
    (ObjectType.CAR, np.array([142, 0, 0])),
    (ObjectType.PEDESTRIAN, np.array([60, 20, 220])),
    (ObjectType.TRAFFIC_LIGHT, np.array([30, 170, 250])),
    (ObjectType.TRAFFIC_SIGN, np.array([0, 220, 220])),
    (ObjectType.BUS, np.array([100, 60, 0])),
    (ObjectType.MOTOR_CYCLE, np.array([230, 0, 0])),
    (ObjectType.BICYLE, np.array([32, 11, 119])),
    (ObjectType.TRUCK, np.array([70, 0, 0])),
    (ObjectType.TRAIN, np.array([100, 80, 0])),
    (ObjectType.RIDER, np.array([0, 0, 255])),
]


def extract_objects(
    segmentation_image: np.ndarray,
) -> List[Tuple[ObjectType, BoundingBox]]:
    """Extracts a list of object labels and bounding boxes from a segmentation images"""
    global mappings
    result_objects: List[Tuple[ObjectType, BoundingBox]] = []
    for type, mask_color in mappings:
        boxes = bounding_boxes_for_mask(segmentation_image, mask_color)
        result_objects.extend([(type, box) for box in boxes])
    return result_objects


def bounding_boxes_for_mask(
    segmentation_image: np.ndarray, mask_color: np.ndarray
) -> List[BoundingBox]:
    mask = np.zeros(segmentation_image.shape[:-1], dtype=np.uint8)
    mask[
        (segmentation_image[:, :, 0] == mask_color[0])
        & (segmentation_image[:, :, 1] == mask_color[1])
        & (segmentation_image[:, :, 2] == mask_color[2])
    ] = 255

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < 50:  # Filter small contours
            continue

        # Get bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append(BoundingBox(x, x + w, y, y + h))
    return boxes


if __name__ == "__main__":
    segmentation_image = cv2.imread("test_segmentation_image.jpg")
    result = extract_objects(segmentation_image)
    print(result)
    result_as_objects = [
        Object(type=type, boundingBox=box, confidence=1, angle=0, distance=0)
        for type, box in result
    ]

    output_image = add_object_information(segmentation_image, result_as_objects)
    cv2.imwrite("test_segmentation_image_output.jpg", output_image)
