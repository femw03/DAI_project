import os
from typing import Dict, List

import torch
from lightning import Trainer
from torch.utils.data import DataLoader
from ultralytics import YOLO

from ..interfaces import (
    BoundingBox,
    CVBridge,
    Image,
    Lidar,
    Object,
    ObjectType,
    ProcessingFinishedCallBack,
)
from .calculate_distance import calculate_anlge, calculate_object_distance
from .traffic_sign_classification import TRANSFORM, CNNModel, ImageDataset, TrafficSign


class ComputerVisionModule(CVBridge):
    def __init__(
        self,
        onProcessingFinished: ProcessingFinishedCallBack,
        FOV: float,
    ) -> None:
        super().__init__(onProcessingFinished)
        self.FOV = FOV
        """The Field Of View in degrees"""
        # Load in models
        current_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "weights"
        )
        self.big_net = YOLO(os.path.join(current_dir, "big_net.pt"), task="detect")
        self.traffic_sign_classifier = CNNModel(
            model=torch.load(os.path.join(current_dir, "traffic_sign.pth"))
        )
        self.lightning_trainer = Trainer(logger=False)

    def on_data_received(
        self, image: Image, lidar: Lidar, current_speed: float
    ) -> None:
        results = self.big_net.predict(
            image.get_image_bytes(), half=True, device="cuda:0"
        )

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
                lidar.get_lidar_bytes(), bounding_box
            )
            object_angle = calculate_anlge(
                object_location.location[0], self.FOV, lidar.get_lidar_bytes().shape[1]
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
        # traffic_signs = [
        #     detected_object
        #     for detected_object in detected
        #     if detected_object.type == ObjectType.TRAFFIC_SIGN
        # ]
        # traffic_signs = self.classifiy_traffic_sign(traffic_signs, image)

        self.submitObjects(detected, image)

    def classifiy_traffic_sign(
        self, traffic_signs: List[Object], image: Image
    ) -> List[TrafficSign]:
        """For each traffic sign in the list, snip it out of the image and pass it to the classifier network"""
        image_bytes = image.get_image_bytes()
        bounding_boxes = [traffic_sign.boundingBox for traffic_sign in traffic_signs]
        image_snips = [
            image_bytes[box.x1 : box.x2, box.y1 : box.y2, :] for box in bounding_boxes
        ]
        loader = DataLoader(
            ImageDataset(data=image_snips, transform=TRANSFORM),
            batch_size=len(traffic_signs),
        )
        results = self.lightning_trainer.predict(self.traffic_sign_classifier, loader)
        return results

    def _submitObjects(self, objects: List[Object], image: Image) -> None:
        pass  # TODO
