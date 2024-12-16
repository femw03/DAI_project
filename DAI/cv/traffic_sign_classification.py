"""
This file contains all code pertaining to traffic SIGN classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ..interfaces import Image, Object

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
"""The transform that need be used for the classifier, is the same as the one it was trained on"""


class ImageDataset(Dataset):
    """A custom dataset to handle in memory numpy image representations"""

    def __init__(self, data: list[np.ndarray], transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            x = PILImage.fromarray(self.data[index].astype(np.uint8))
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.data)


@dataclass
class TrafficSign:
    """A class containing the traffic sign classification result"""

    type: TrafficSignType
    confidence: float

    @staticmethod
    def cspeed_limit(traffic_signs: List[TrafficSign]) -> Optional[float]:
        """Find the speed limit from the traffic signs"""
        valid_speeds = [
            traffic_sign.type.speed_limit
            for traffic_sign in traffic_signs
            if traffic_sign.type.speed_limit is not None and traffic_sign.confidence >= 0.6
        ]
        return min(valid_speeds) if len(valid_speeds) != 0 else None


class TrafficSignType(Enum):
    """An enum containing all Traffic sign classification types and their encoded meaning"""

    BACK = ("back", 0, None)
    SPEED_30 = ("speed_30", 1, 30)
    SPEED_A_30 = ("speed_limit_30", 4, 30)
    SPEED_A_40 = ("speed_limit_40", 5, 40)
    SPEED_60 = ("speed_60", 2, 60)
    SPEED_A_60 = ("speed_limit_60", 6, 60)
    SPEED_90 = ("speed_90", 3, 90)
    STOP = ("stop", 7, None)

    def __init__(self, label: str, index: int, speed_limit: Optional[int]) -> None:
        super().__init__()
        self._label = label
        """The classification label"""
        self.index = index
        """The index of the classification label"""
        self.speed_limit = speed_limit
        """The speed limit associatied with the sign, None if it is not relevant"""

    @classmethod
    def from_index(cls, index: int):
        """Get the enum from their classifier label index"""
        for sign in cls:
            if sign.index == index:
                return sign
        raise ValueError(f"No TrafficSignClass with index {index} found.")


class ConvolutionalClassifier(nn.Module):
    """
    A general description of the convolutional network for classification
    side note:
        This is seperated from CNNModel because it was trained without pytorch which resulted
        in the weight file being incompatible.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Adjusted for increased capacity
        self.fc2 = nn.Linear(128, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # Conv2D -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # Flatten
        x = F.relu(self.fc1(x))  # Dense layer
        x = self.fc2(x)  # Output layer with softmax activation
        return F.softmax(x, dim=1)


class TrafficSignClassifier(L.LightningModule):
    """
    A pytorch lightning module for traffic_sign classification,
    use torch.load to load in the correct ConvolutionalClassifier (see note).
    only the predict step is implemented
    """

    @staticmethod
    def from_weights_file(weights_file: str) -> TrafficSignClassifier:
        model_dict = torch.load(weights_file, weights_only=True)
        model = ConvolutionalClassifier()
        model.load_state_dict(model_dict)
        return TrafficSignClassifier(model)

    def __init__(self, model: ConvolutionalClassifier):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input an image as a torch tensor and returns a probability distribution according TrafficSignClass"""
        return self.model(x)

    def predict_step(
        self, batch: torch.Tensor, batch_index: torch.Tensor
    ) -> List[TrafficSignType]:
        """batch has shape (batch_size, img_width, img_height) with the image dimension being 32x32 and for each image in the batch it returns it's TrafficSignClass"""
        probabilities = self.forward(batch)  # (batch_size, num_classes)
        prediction_value, prediction_index = probabilities.max(dim=1)  # (batch_size, 1)
        prediction_classes = [
            TrafficSignType.from_index(index.item()) for index in prediction_index
        ]
        result = [
            TrafficSign(type, value)
            for type, value in zip(prediction_classes, prediction_value)
        ]
        return result

    @staticmethod
    def classify(
        trainer: L.Trainer,
        model: TrafficSignClassifier,
        traffic_signs: List[Object],
        image: Image,
    ) -> List[TrafficSign]:
        """For each traffic sign in the list, snip it out of the image and pass it to the classifier network"""
        if len(traffic_signs) == 0:
            return []
        image_bytes = image.get_image_bytes()
        bounding_boxes = [traffic_sign.boundingBox for traffic_sign in traffic_signs]
        image_snips = [
            image_bytes[box.y1 : box.y2, box.x1 : box.x2, :] for box in bounding_boxes
        ]
        loader = DataLoader(
            ImageDataset(data=image_snips, transform=TRANSFORM),
            batch_size=len(traffic_signs),
        )
        results = trainer.predict(model, loader)
        return results[
            0
        ]  # trainer.predict returns a list of results for every loader that is passed as an argument
