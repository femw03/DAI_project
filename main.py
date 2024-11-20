import logging
from typing import List

import cv2
import numpy as np

from DAI.cv import ComputerVisionModule
from DAI.interfaces.interfaces import Image, Object
from DAI.simulator import NumpyImage, NumpyLidar, World

font = cv2.FONT_HERSHEY_SIMPLEX


def onImageReceived(*args) -> None:
    print("recieved data")


def onProcessingFinished(objects: List[Object], image: Image) -> None:  # noqa: F821
    print("finished processing")
    image_bytes = image.get_image_bytes()
    for object in objects:
        box = object.boundingBox
        cv2.rectangle(
            image_bytes, (box.x1, box.y1), (box.x2, box.y2), color=(255, 0, 0)
        )
        cv2.putText(
            image_bytes,
            f"{object.type}\n{object.angle}",
            (box.x2, box.y1),
            font,
            0.25,
            (255, 0, 0),
            1,
        )
    cv2.imwrite("out.png", image_bytes)


logging.basicConfig(level=logging.INFO)
world = World(onImageReceived=onImageReceived)
print(world)

cv = ComputerVisionModule(onProcessingFinished=onProcessingFinished, FOV=world.view_FOV)
print(cv)
image = NumpyImage(cv2.imread("image11.png"))
lidar = NumpyLidar(np.empty_like(image.get_image_bytes())[:, :, 0])
cv.on_data_received(image=image, lidar=lidar, current_speed=0)
# world.start()
