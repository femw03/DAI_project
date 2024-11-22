import os
from typing import Dict, List

import torch
from lightning import Trainer
from ultralytics import YOLO

from ..interfaces import (
    BoundingBox,
    CarlaData,
    CarlaFeatures,
    ComputerVisionModule,
    Object,
    ObjectType,
)
from .calculate_distance import calculate_anlge, calculate_object_distance
from .traffic_sign_classification import TrafficSign, TrafficSignClassifier


class ComputerVisionModuleImp(ComputerVisionModule):
    """
    Processes the data using the following strategy:
    1. Use big_net to detect every interesting object available
    2. For all detected traffic_signs use the traffic_sign_classifier to classifiy the signs
        a. Use the classified list of traffic signs to extract the current maximum speed
    3. For all detected traffic_light use the traffic_light classifier to detect their relevance and color value
        # TODO (implement)
    """

    def __init__(
        self,
    ) -> None:
        # Load in models
        current_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "weights"
        )
        self.big_net = YOLO(os.path.join(current_dir, "big_net.pt"), task="detect")
        self.traffic_sign_classifier = TrafficSignClassifier(
            model=torch.load(os.path.join(current_dir, "traffic_sign.pth"))
        )
        self.lightning_trainer = Trainer(logger=False)

    def process_data(self, data: CarlaData) -> CarlaFeatures:
        # Use the big net to detect objects generally
        results = self.big_net.predict(
            data.rgb_image.get_image_bytes(), half=True, device="cuda:0"
        )

        # Convert the YOLO Results to a list of Objects
        detected: List[Object] = []
        result = results[0]
        probabilities: List[float] = result.boxes.conf.tolist()
        label_indices: List[int] = result.boxes.cls.tolist()
        labels: Dict[int, str] = result.names
        bounding_box_coords: List[List[float]] = result.boxes.xyxy.to(
            torch.int
        ).tolist()

        for confidence, label_index, bounding_box_coord in zip(
            probabilities, label_indices, bounding_box_coords
        ):
            type = ObjectType(labels[label_index])
            bounding_box = BoundingBox.from_array(bounding_box_coord)
            object_location = calculate_object_distance(
                data.lidar_data.get_lidar_bytes(), bounding_box
            )
            object_angle = calculate_anlge(
                object_location.location[0],
                self.FOV,
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

        # Use the traffic sign classifier to get more details about the traffic signs
        traffic_signs = [
            detected_object
            for detected_object in detected
            if detected_object.type == ObjectType.TRAFFIC_SIGN
        ]
        traffic_signs = TrafficSignClassifier.classify(
            self.lightning_trainer,
            self.traffic_sign_classifier,
            traffic_signs,
            data.rgb_image,
        )
        max_speed = TrafficSign.speed_limit(traffic_signs)

        return CarlaFeatures(
            objects=detected,
            current_speed=data.current_speed,
            max_speed=max_speed,
        )
