"""This file converts a present observation into the features we require."""

from __future__ import annotations

from typing import Optional

import torch

from ..interfaces import AgentFeatures, CarlaObservation, FeatureExtractor
from ..simulator import CarlaWorld
from ..simulator.extract import (
    find_vehicle_in_front,
    get_current_max_speed,
    get_current_speed,
    get_distance_to_leading,
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
    ):
        super().__init__()
        self.current_speed = current_speed  # km/s
        self.max_speed = max_speed  # km/s
        self.is_car_in_front = is_car_in_front
        self.distance_to_car_in_front = distance_to_car_in_front

    def to_tensor(self) -> torch.FloatTensor:
        return torch.tensor(
            [
                self.current_speed / MAX_SPEED,
                self.max_speed / MAX_SPEED,
                1 if self.is_car_in_front else 0,
                MAX_DISTANCE
                if self.distance_to_car_in_front is None
                else self.distance_to_car_in_front / MAX_DISTANCE,
            ],
            dtype=torch.float32,
        )


class SimpleFeatureExtractor(FeatureExtractor):
    def __init__(self, width, margin, correction_factor, boost_factor):
        super().__init__()
        self.previous_max_speed = None
        self.width = width
        self.margin = margin
        self.correction_factor = correction_factor
        self.boost_factor = boost_factor

    def extract(self, observation: CarlaObservation) -> SimpleFeatures:
        current_speed = observation.current_speed
        observed_max_speed = observation.max_speed
        if observed_max_speed is not None:
            self.previous_max_speed = observed_max_speed
            max_speed = observed_max_speed
        else:
            if self.previous_max_speed is None:
                max_speed = MAX_SPEED
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
            distance_to_vehicle_front = vehicle_in_front.distance
        else:
            distance_to_vehicle_front = None
        return SimpleFeatures(
            current_speed=current_speed,
            max_speed=max_speed,
            is_car_in_front=is_vehicle_in_front,
            distance_to_car_in_front=distance_to_vehicle_front,
        )

    def reset(self):
        self.previous_max_speed = None


def get_perfect_obs(world: CarlaWorld) -> CarlaObservation:
    object_list = get_objects(world)
    speed_limit = get_current_max_speed(world)
    current_speed = get_current_speed(world)

    return CarlaObservation(
        objects=object_list,
        current_speed=current_speed,
        max_speed=speed_limit,
        angle=get_steering_angle(world),
        distance_to_pedestrian_crossing=None,
        distance_to_stop=None,
        pedestrian_crossing_flag=None,
        stop_flag=None,
    )
    # traffic_light = get_current_affecting_light_state(world)
    # distance_to_stop = None
    # distance_to_crossing = None

    # Make Stop Flag
    # if (
    #     traffic_light == CarlaTrafficLightState.RED
    #     or traffic_light == CarlaTrafficLightState.YELLOW
    # ):
    #     stop_flag = 1
    # else:
    #     stop_flag = 0

    # Remove traffic lights and traffic signs and normalize distance
    # filtered_objects = []
    # for obj in object_list:
    #     if (
    #         obj.type not in [ObjectType.TRAFFIC_LIGHT, ObjectType.TRAFFIC_SIGN]
    #         and obj.distance <= self.relevant_distance + 1
    #     ):
    #         obj.distance = min(1, obj.distance / self.relevant_distance)
    #         if obj.type == ObjectType.PEDESTRIAN:
    #             obj.type = 0  # pedestrians
    #         else:
    #             obj.type = 1  # vehicles, bicycles, motorcycles
    #         filtered_objects.append(obj)
    # # Sort by decreasing distance level
    # filtered_objects.sort(
    #     key=lambda obj: obj.distance
    # )  # because confidence = 1 (perfect information)
    # # Truncate list to have the size of max_objects
    # if len(filtered_objects) > self.max_objects:
    #     filtered_objects = filtered_objects[: self.max_objects]
    # else:
    #     # Perform zero padding
    #     padding_needed = self.max_objects - len(filtered_objects)
    #     padding = [
    #         Object(type=random.randint(0, 1), confidence=0.0, distance=0.0, angle=0.0)
    #         for _ in range(padding_needed)
    #     ]
    #     filtered_objects.extend(padding)

    # logger.info(f"Object list: {object_list}")
    # logger.info(f"Filtered objects observation: {filtered_objects}")
    # logger.info(f"Speed limit: {speed_limit}")
    # logger.info(f"Current speed: {current_speed}")
    # logger.info(f"Stop flag: {stop_flag}")

    # Static features (speed_limit, current_speed)
    # static_features = np.array([speed_limit, current_speed], dtype=np.float32)

    # # Task features
    # task_features = np.array(
    #     [stop_flag, 0, 0, 0], dtype=np.float32
    # )  # Placeholder for "DistanceToStopLine", "CrossingDetected", "DistanceToCrossing"

    # # Object features
    # object_features = np.array(
    #     [
    #         [obj.type, obj.confidence, obj.distance, obj.angle]
    #         for obj in filtered_objects
    #     ],
    #     dtype=np.float32,
    # )

    # # Flatten the entire observation into a single vector
    # observation = np.concatenate(
    #     [static_features, task_features, object_features.flatten()]
    # )

    # logger.info(f"Observation: {observation}")

    # return observation
