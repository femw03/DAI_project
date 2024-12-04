"""
This file is a script where one can quickly manually validate if the cv implementation
is working correctly without depending on the carla simulator or agent.
"""

import argparse
import sys

import cv2
import numpy as np
from loguru import logger

from ..cv import ComputerVisionModuleImp
from ..interfaces import CarlaData, ObjectType
from ..simulator import NumpyImage, NumpyLidar
from ..utils import timeit
from ..visuals.visual_utils import add_object_information, add_static_information


def main(image_path: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    logger.info(f"Reading image {image_path}")
    image = cv2.imread(image_path)
    data = CarlaData(
        rgb_image=NumpyImage(image, 90),
        lidar_data=NumpyLidar(
            np.random.random(image.shape[:2]), 90, lambda x: x * 1000
        ),
        current_speed=0,
    )
    cv = ComputerVisionModuleImp()

    logger.info("Processing image")
    # Warmup
    cv.process_data(data)

    # Real
    result, time = timeit(lambda: cv.process_data(data))
    logger.info(f"Finished in {time:.3}s")
    logger.debug(result)
    image_with_bb = add_object_information(image, result.objects)
    image_with_static = add_static_information(image_with_bb, result, time)

    logger.info("Writing result to out.png")
    cv2.imwrite("out.png", image_with_static)
    traffic_signs = [
        detected_object
        for detected_object in result.objects
        if detected_object.type == ObjectType.TRAFFIC_SIGN
    ]
    logger.info(f"Found {len(traffic_signs)} traffic signs")
    for i, traffic_sign in enumerate(traffic_signs):
        box = traffic_sign.boundingBox
        try:
            cv2.imwrite(
                f"test/{i}_{traffic_sign.type.name}.png",
                image[box.y1 : box.y2, box.x1 : box.x2, :],
            )
        except Exception:
            logger.error("failed writing traffic sign snippet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-image",
        "-i",
        help="Path to the input image",
        default="test.png",
        required=False,
    )
    args = parser.parse_args()
    main(args.input_image)