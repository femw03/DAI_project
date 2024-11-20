import sys
import time
from typing import List

import cv2
import numpy as np
from loguru import logger

from DAI.cv import ComputerVisionModule
from DAI.interfaces.interfaces import Image, Object
from DAI.simulator import NumpyImage, NumpyLidar, World
from DAI.visuals import Visuals

font = cv2.FONT_HERSHEY_SIMPLEX
logger.remove()
logger.add(sys.stderr, level="INFO")


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


visuals = Visuals(640, 480, 30, on_quit=quit)
world = World(
    on_rgb_received=visuals.set_rgb_image,
    on_image_received=lambda *args: print("data for agent"),
    view_height=visuals.height,
    view_width=visuals.width,
)


visuals.on_quit = world.stop

world.start()

visuals.start()

world.join()
visuals.join()
