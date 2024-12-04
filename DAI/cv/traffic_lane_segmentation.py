import os
from typing import Tuple

import lightning as L
import torch
import torch.nn as nn
from PIL import Image
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torchvision.transforms import transforms


class ImageLabelDataset(Dataset):
    def __init__(self, root_dir, transform=transforms.ToTensor()):
        self.images_dir = os.path.join(root_dir, "images")
        self.labels_dir = os.path.join(root_dir, "labels")
        self.image_files = sorted(os.listdir(self.images_dir))
        self.label_files = sorted(os.listdir(self.labels_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        # Load label
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        label = Image.open(label_path).convert("1")

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label


class TrafficLaneSegmentationDataModule(L.LightningDataModule):
    def __init__(self, root_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Transforms
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def setup(self, stage=None):
        """
        Called to initialize the datasets.
        """
        self.dataset = ImageLabelDataset(self.root_dir, transform=self.transform)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [0.8, 0.1, 0.1]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


class TrafficLaneSegmentationModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        # Use MobileNetV2 as backbone for real-time performance
        self.model = lraspp_mobilenet_v3_large(num_classes=1)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)["out"]

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters())

    def training_step(
        self, batch: Tuple[torch.FloatTensor, torch.IntTensor], batch_idx
    ):
        images, masks = batch
        # Ensure masks are float and same shape as output
        outputs = self.forward(images)
        loss = self.loss(outputs, masks)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: Tuple[torch.FloatTensor, torch.IntTensor], batch_idx
    ):
        images, masks = batch
        outputs = self(images)
        loss = self.loss(outputs, masks)
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch: torch.FloatTensor, batch_idx):
        images = batch
        return torch.sigmoid(self(images)) > 0.5
