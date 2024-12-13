"""
This file contains the code to extract perfect information from the carla world
"""

from typing import List, Optional

import numpy as np
from loguru import logger

from ..cv.calculate_distance import calculate_anlge, calculate_object_distance
from ..interfaces import Object, ObjectType
from .carla_world import CarlaWorld
from .numpy_image import NumpyLidar
from .segmentation import extract_objects
from .wrappers import CarlaTrafficLightState, CarlaVector3D


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
            confidence=1.0,  # added
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
    return world.car.velocity.magnitude * 3.6


def get_current_affecting_light_state(world: CarlaWorld) -> CarlaTrafficLightState:
    """
    Get the state of the traffic light that is affecting the vehicle,
    If no lights are affecting the car the state will be green.
    """
    return world.car.get_traffic_light_state


def has_collided(world: CarlaWorld) -> bool:
    """Returns true if the vehicle has collided with another actor"""
    return world.collision is not None


def get_steering_angle(world: CarlaWorld) -> float:
    if len(world.local_planner.get_plan()) != 0:
        next_wp, _ = world.local_planner.get_plan()[0]
    else:
        return 0
    next_location = next_wp.location
    current_location = world.car.location
    desired_direction_vector = current_location.vector_to(next_location)
    car_forward_vector = CarlaVector3D(world.car.transform.get_forward_vector())
    return car_forward_vector.angle_to(desired_direction_vector)


def find_vehicle_in_front(
    angle: float, observation: List[Object], threshold=np.pi / 36
) -> Optional[Object]:
    """Finds the vehicle in front of the car,"""
    vehicle_types = [
        ObjectType.BICYLE,
        ObjectType.BUS,
        ObjectType.CAR,
        ObjectType.MOTOR_CYCLE,
        ObjectType.RIDER,
        ObjectType.TRAILER,
        ObjectType.TRUCK,
    ]

    vehicles = [obj for obj in observation if obj.type in vehicle_types]
    within_angle = [
        (vehicle, abs(vehicle.angle - angle))
        for vehicle in vehicles
        if abs(vehicle.angle - angle) < threshold
    ]
    sorted_by_distance = sorted(
        within_angle, key=lambda vehicle: vehicle[0].distance / 20 + vehicle[1]
    )
    return sorted_by_distance[0][0] if len(sorted_by_distance) > 0 else None


def has_completed_navigation(world: CarlaWorld):
    return world.local_planner.done()

def get_distance_to_leading(world: CarlaWorld):
    return world.car.location.distance_to(world.lead_car.location)
