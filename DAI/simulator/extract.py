"""
This file contains the code to extract perfect information from the carla world
"""

from typing import List

from loguru import logger

from ..cv.calculate_distance import calculate_anlge, calculate_object_distance
from ..interfaces import Object
from .carla_world import CarlaWorld
from .numpy_image import NumpyLidar
from .segmentation import extract_objects
from .wrappers import CarlaTrafficLightState


def get_objects(world: CarlaWorld) -> List[Object]:
    """
    Get a list of object from the carla world instance,
    be aware that the depth and segmentation need to be available
    """
    # Step 1 get segmentation image and extract objects
    segmentation_image = world.segm_image
    if segmentation_image is None:
        logger.warning(
            "Tried to get object information from carla but no segmentation image was available"
        )
        return []

    result = extract_objects(segmentation_image)

    # Step 2 get depth and and add to results
    depth = NumpyLidar(
        world.depth_image,
        world.view_FOV,
        lambda x: (x / 255) * 1000,
    )
    result_with_spatial_info = [
        (type, box, calculate_object_distance(depth, box)) for type, box in result
    ]

    # Convert to objects
    objects = [
        Object(
            type=type,
            boundingBox=box,
            distance=distance_info.depth,
            angle=calculate_anlge(
                distance_info.location[0], world.view_FOV, world.view_width
            ),
        )
        for type, box, distance_info in result_with_spatial_info
    ]
    return objects


def get_current_max_speed(world: CarlaWorld) -> float:
    """Get the max speed affecting the current vehicle"""
    return world.car.current_max_speed


def get_current_speed(world: CarlaWorld) -> float:
    """Get the current speed of the vehicle"""
    return world.car.velocity


def get_current_affecting_light_state(world: CarlaWorld) -> CarlaTrafficLightState:
    """
    Get the state of the traffic light that is affecting the vehicle,
    If no lights are affecting the car the state will be green.
    """
    return world.car.get_traffic_light_state


def has_collided(world: CarlaWorld) -> bool:
    """Returns true if the vehicle has collided with another actor"""
    return world.collision is not None
