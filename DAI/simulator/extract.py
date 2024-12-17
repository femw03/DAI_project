"""
This file contains the code to extract perfect information from the carla world
"""

from typing import List, Optional

from loguru import logger

from ..cv.calculate_distance import calculate_anlge, calculate_object_distance
from ..cv.lane_detection import expected_deviation
from ..interfaces import Object, ObjectType
from .carla_world import CarlaWorld
from .numpy_image import NumpyLidar
from .segmentation import extract_objects
from .wrappers import (
    CarlaTrafficLight,
    CarlaTrafficLightState,
    CarlaVector3D,
    CarlaWaypoint,
)


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
        lambda x: x * 1000,
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
    angle: float,
    observation: List[Object],
    width,
    threshold,
    correction_factor,
    boost_factor,
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
    expected_vehicle_deviations = [
        expected_deviation(
            vehicle.distance,
            angle,
            correction_factor=correction_factor,
            boost_factor=boost_factor,
        )
        for vehicle in vehicles
    ]
    margins = [(threshold / (vehicle.distance + 2)) for vehicle in vehicles]
    vehicle_x_coords = [
        (vehicle.boundingBox.x2 - vehicle.boundingBox.x1) // 2 + vehicle.boundingBox.x1
        for vehicle in vehicles
    ]
    centered_x_coords = [
        vehicle_x_coord - (width // 2) for vehicle_x_coord in vehicle_x_coords
    ]

    valid_vehicles = [
        vehicle
        for vehicle, deviation, margin, x_courd in zip(
            vehicles, expected_vehicle_deviations, margins, centered_x_coords
        )
        if abs(deviation - x_courd) < margin
    ]
    # print(
    #     expected_vehicle_deviations,
    #     margins,
    #     # vehicle_x_coords,
    #     centered_x_coords,
    #     [vehicle.distance for vehicle in vehicles],
    #     # valid_vehicles,
    # )
    sorted_by_distance = sorted(valid_vehicles, key=lambda vehicle: vehicle.distance)
    return sorted_by_distance[0] if len(sorted_by_distance) > 0 else None


def has_completed_navigation(world: CarlaWorld):
    return world.local_planner.done(world.car.location)


def get_distance_to_leading(world: CarlaWorld):
    return world.car.location.distance_to(world.lead_car.location)


def get_stop_point(world: CarlaWorld, traffic_light: CarlaTrafficLight) -> Optional[CarlaWaypoint]:
    """if a traffic light is affecting the car return the distance to it's supposed stop_point"""
    stop_points = traffic_light.stop_points
    route = [wp for wp, _ in world.local_planner.get_plan()]
    if len(stop_points) == 0:
        return None
    # Problem we need to find the stop point that is on our route
    # Strategy 1 determine for each waypoint if it comes within 1 meter of the route and at what index
    for wp in route:
        for stop_wp in stop_points:
            distance = wp.location.distance_to(stop_wp.location)
            if distance < 0.5:  # If within a meter of the stop point it must coincide
                return stop_wp

    # None of the stop points came close enough to the route this might be because the planned route ends
    # prematurely, then we decide that the closest wp must be the waypoint
    distances = [
        (world.car.location.distance_to(stop_wp.location), stop_wp)
        for stop_wp in stop_points
    ]
    return min(distances, key=lambda dist: dist[0])[1]


def get_affecting_traffic_lightV2(world: CarlaWorld, traffic_lights: List[CarlaTrafficLight], route: List[CarlaWaypoint]) -> Optional[CarlaTrafficLight]:
    for waypoint in route:
        for traffic_light in traffic_lights:
            for affected_waypoint in traffic_light.affected_waypoints:
                distance = world.car.location.distance_to(affected_waypoint.location)
                if distance > 40: 
                    break
                delta = waypoint.location.distance_to(affected_waypoint.location)
                if delta < 0.5:
                    return traffic_light
    return None
