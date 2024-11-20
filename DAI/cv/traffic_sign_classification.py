from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
"""The transform that need be used for the classifier"""


@dataclass
class TrafficSign:
    type: TrafficSignType
    confidence: float


class TrafficSignType(Enum):
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
        for sign in cls:
            if sign.index == index:
                return sign
        raise ValueError(f"No TrafficSignClass with index {index} found.")


class ConvolutionalClassifier(nn.Module):
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
        return F.log_softmax(x, dim=1)


class CNNModel(L.LightningModule):
    def __init__(self, model: ConvolutionalClassifier):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input an image as a torch tensor and returns a probability distribution according TrafficSignClass"""
        return self.model(x)

    def predict_step(
        self, batch: torch.Tensor, batch_index: torch.Tensor, dataloader_index: int
    ) -> List[TrafficSignType]:
        """batch has shape (batch_size, img_width, img_height) with the image dimension being 32x32 and for each image in the batch it returns it's TrafficSignClass"""
        probabilities = self.forward(batch)  # (batch_size, num_classes)
        prediction_index = torch.argmax(probabilities, dim=1)  # (batch_size, 1)
        prediction_value = probabilities.max(dim=1)  # (batch_size, 1)
        prediction_classes = [
            TrafficSignType.from_index(index.item()) for index in prediction_index
        ]
        result = [
            TrafficSign(type, value)
            for type, value in zip(prediction_classes, prediction_value)
        ]
        return result


class ImageDataset(Dataset):
    def __init__(self, data: list[np.ndarray], transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1, 2, 0))
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.data)
