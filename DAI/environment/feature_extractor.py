"""This file converts a present observation into the features we require."""

from __future__ import annotations

from typing import Optional

import torch

from ..interfaces import AgentFeatures, CarlaObservation, FeatureExtractor
from ..simulator import CarlaWorld, tracker, wrappers
from ..simulator.extract import (
    find_vehicle_in_front,
    get_current_max_speed,
    get_current_speed,
    # get_distance_to_leading,
    get_objects,
    get_steering_angle,
)

MAX_SPEED = 120.0  # km/s
MAX_DISTANCE = 100.0  # m


class SimpleFeatures(AgentFeatures):
    def __init__(
        self,
        current_speed: float,
        max_speed: float,
        is_car_in_front: bool,
        distance_to_car_in_front: Optional[float],
        should_stop: bool,
        distance_to_stop: Optional[float],
    ):
        super().__init__()
        self.current_speed = current_speed  # km/s
        self.max_speed = max_speed  # km/s
        self.is_car_in_front = is_car_in_front
        self.distance_to_car_in_front = distance_to_car_in_front
        self.should_stop = should_stop
        self.distance_to_stop = distance_to_stop

    def to_tensor(self) -> torch.FloatTensor:
        return torch.tensor(
            [
                self.current_speed / MAX_SPEED,
                self.max_speed / MAX_SPEED,
                1 if self.is_car_in_front else 0,
                0
                if self.distance_to_car_in_front is None
                else self.distance_to_car_in_front / MAX_DISTANCE,
                1 if self.should_stop else 0,
                0
                if self.distance_to_stop is None
                else self.distance_to_stop / MAX_DISTANCE,
            ],
            dtype=torch.float32,
        )


class SimpleFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        width,
        margin,
        correction_factor,
        boost_factor,
        traffic_light_distance_bias: float,
    ):
        super().__init__()
        self.previous_max_speed = None
        self.width = width
        self.margin = margin
        self.correction_factor = correction_factor
        self.boost_factor = boost_factor
        self.traffic_light_distance_bias = traffic_light_distance_bias
        self.previous_distance = 0

    def extract(self, observation: CarlaObservation) -> SimpleFeatures:
        current_speed = observation.current_speed
        observed_max_speed = observation.max_speed
        if observed_max_speed is not None:
            self.previous_max_speed = observed_max_speed
            max_speed = observed_max_speed
        else:
            if self.previous_max_speed is None:
                max_speed = 30  # MAX_SPEED
            else:
                max_speed = self.previous_max_speed

        vehicle_in_front = find_vehicle_in_front(
            observation.angle,
            observation.objects,
            width=self.width,
            threshold=self.margin,
            correction_factor=self.correction_factor,
            boost_factor=self.boost_factor,
        )
        is_vehicle_in_front = vehicle_in_front is not None
        if is_vehicle_in_front:
            self.previous_distance = vehicle_in_front.distance
            distance_to_vehicle_front = vehicle_in_front.distance
        else:
            distance_to_vehicle_front = self.previous_distance
        return SimpleFeatures(
            current_speed=current_speed,
            max_speed=max_speed,
            is_car_in_front=is_vehicle_in_front,
            distance_to_car_in_front=distance_to_vehicle_front,
            distance_to_stop=observation.distance_to_stop - self.traffic_light_distance_bias
            if observation.distance_to_stop is not None
            else None,
            should_stop=observation.red_light,
        )

    def reset(self):
        self.previous_max_speed = None


def get_perfect_obs(world: CarlaWorld) -> CarlaObservation:
    object_list = get_objects(world)
    speed_limit = get_current_max_speed(world)
    current_speed = get_current_speed(world)
    angle = get_steering_angle(world)
    route = [waypoint for waypoint, _ in world.local_planner.get_plan()]
    next_wp_result = tracker.find_next_wp_from(route, min_distance=20)
    if next_wp_result is not None:
        angle = wrappers.CarlaVector3D(
            world.car.transform.get_forward_vector()
        ).angle_to(world.car.location.vector_to(next_wp_result[0].location))

    return CarlaObservation(
        objects=object_list,
        current_speed=current_speed,
        max_speed=speed_limit,
        angle=angle,
        distance_to_pedestrian_crossing=None,
        distance_to_stop=None,
        pedestrian_crossing_flag=None,
        red_light=None,
    )
    