from typing import Any, Dict

import gymnasium as gym

# import keyboard
import numpy as np

import wandb

from ..cv import ComputerVisionModuleImp
from ..simulator import CarlaWorld, tracker, wrappers
from ..simulator.extract import (
    find_vehicle_in_front,
    get_current_max_speed,
    get_current_speed,
    get_distance_to_leading,
    get_objects,
    get_steering_angle,
    has_collided,
    has_completed_navigation,
)
from ..visuals import ObjectDTO, Visuals
from .carla_setup import setup_carla
from .feature_extractor import SimpleFeatureExtractor


class CarlaEnv2(gym.Env):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.collisionCounter = 0
        self.config = config
        self.perfect = config["perfect"]
        self.world_max_speed = config["world_max_speed"]
        self.visuals = Visuals(fps=30, width=1280, height=720)
        self.world: CarlaWorld = setup_carla(self.visuals)
        self.episode_reward = 0
        self.episode = 0
        self.detected_distance = 0
        self.detected_speed_limit = 0
        self.detected_vehicle = 0
        self.last_speed = 0
        # self.wrongSpeedLimitCounter = 0

        self.max_objects = config[
            "max_objects"
        ]  # Maximum number of objects per observation
        self.relevant_distance = config["relevant_distance"]

        self.collisionFlag = False
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array(
                [
                    self.world_max_speed + 10,
                    self.world_max_speed + 10,
                    1,
                    self.relevant_distance + 10,
                    1,
                    110,
                ]
            ),
            dtype=np.float32,
        )
        # Define actions space
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.action_taken = 0
        self.feature_extractor = SimpleFeatureExtractor(
            margin=self.visuals.margin,
            width=self.visuals.width,
            correction_factor=self.visuals.correction_factor,
            boost_factor=self.visuals.boost_factor,
        )
        self.should_stop = False
        self.distance_to_stop = 0

        self.cv = ComputerVisionModuleImp()

    def reset(self, seed=None, **kwargs):
        """
        Reset the environment to initial state and return initial observation.
        """
        print("resetting")
        self.episode += 1
        wandb.log({"episode_reward": self.episode_reward, "episode": self.episode})
        self.episode_reward = 0
        # Reset Carla world
        self.world.reset()
        self.world.await_next_tick()
        # Set the random seed if provided
        self.feature_extractor.reset()
        observation = self.cv.process_data(data=self.world.data)
        observation.max_speed = get_current_max_speed(self.world)
        observation.red_light = None

        self.detected_vehicle = 0
        self.detected_distance = 0
        self.detected_speed_limit = 0
        self.should_stop = False
        self.distance_to_stop = 0
        self.last_speed = 0

        features = self.feature_extractor.extract(observation)
        features.distance_to_stop = 0
        features.should_stop = 0
        return features.to_tensor(), {}

    def step(self, action):
        """
        Apply an action and return the new observation, reward, and done.
        """

        # print("stepping")
        # Apply action
        action = action[0]
        # logger.info(f"Executing step with action {action}")
        self.world.set_speed(action)

        wandb.log({"action": action})
        self.world.await_next_tick()

        # TO DO: wait for execution action in Carla ?
        self.action_taken = action

        # Compute reward
        reward = self._get_reward()
        self.episode_reward += reward
        # Check if the episode is terminated
        if has_completed_navigation(self.world):
            print("completed nav without crash, finding new route!")
            self.world.start_new_route_from_waypoint()

        crash = has_collided(self.world)
        if crash:
            self.collisionCounter += 1
        # dis = False #get_distance_to_leading(self.world) > 75
        dis = get_distance_to_leading(self.world) > 50
        terminated = crash  or dis
        truncated = False
        info = {}

        # observation = get_perfect_obs(world=self.world)  # TODO replace with cv call
        # test TODO
        observation = self.cv.process_data(
            data=self.world.data
        )  # TODO replace with cv call
        # observation.max_speed = get_current_max_speed(self.world)
        # observation.red_light = None  # overwrite for basic training!!!
        self.feature_extractor.margin = self.visuals.margin
        self.feature_extractor.correction_factor = self.visuals.correction_factor
        self.feature_extractor.boost_factor = self.visuals.boost_factor
        features = self.feature_extractor.extract(observation)
        self.detected_speed_limit = features.max_speed
        features.max_speed = get_current_max_speed(self.world)
        features.should_stop = 0
        features.distance_to_stop = 0

        # if self.detected_speed_limit != features.max_speed:
        # self.wrongSpeedLimitCounter += 1

        if features.is_car_in_front:
            self.detected_vehicle = 1
        else:
            self.detected_vehicle = 0

        self.detected_distance = features.distance_to_car_in_front
        # TODO !!!
        # self.should_stop = features.should_stop
        # self.distance_to_stop = features.distance_to_stop
        
        # Let's cheat!!!
        # features.is_car_in_front = True
        # features.distance_to_car_in_front = get_distance_to_leading(self.world)

        return (
            features.to_tensor(),
            reward,
            terminated,
            truncated,
            info,
        )

    def _get_reward(self) -> float:
        """
        Calculate reward based on the action and the current environment state.
        Reward based on speed and distance, with constraints on safe driving.

        Use get_traffic_light function from carla to determine if he ran de stop! 
        Do something like, if current speed > 10 at distance < 1 => kill actor!!!
        """
        if self.world.local_planner.done():
            print(
                "made it to destination in time without crash, lets find another route!"
            )
            self.world.start_new_route_from_waypoint()

        # object_list = get_objects(self.world)
        # plot cv results not perfect.
        observation = self.cv.process_data(self.world.data)
        object_list_imperfect = observation.objects

        speed_limit = get_current_max_speed(self.world)
        current_speed = get_current_speed(self.world)
        # traffic_light = get_current_affecting_light_state(self.world)
        collision = has_collided(self.world)
        angle = get_steering_angle(self.world)
        route = [waypoint for waypoint, _ in self.world.local_planner.get_plan()]
        next_wp_result = tracker.find_next_wp_from(route, min_distance=20)
        if next_wp_result is not None:
            angle = wrappers.CarlaVector3D(
                self.world.car.transform.get_forward_vector()
            ).angle_to(self.world.car.location.vector_to(next_wp_result[0].location))
        self.visuals.angle = angle
        detected_car = find_vehicle_in_front(
            angle,
            object_list_imperfect,  # now we use cv, would like to see results in simulation.
            width=self.visuals.width,
            threshold=self.visuals.margin,
            correction_factor=self.visuals.correction_factor,
            boost_factor=self.visuals.boost_factor,
        )

        # Constants
        speed_margin = 0.1 * speed_limit
        safe_distance_margin = 0.15
        max_safe_distance = 50

        # perfect reward (Leading car)
        # vehicle_in_front = True  # TODO: No car ahead if real world
        # distance_to_car_in_front = get_distance_to_leading(self.world) - 4

        # segmentation reward (No Leading Car)
        if detected_car is not None:
            vehicle_in_front = True
            distance_to_car_in_front = detected_car.distance
        else:
            vehicle_in_front = False
            distance_to_car_in_front = (
                max_safe_distance  # normally not used but to be safe
            )

        # Default reward
        reward = 0

        # Collision handling
        if collision:
            self.collisionFlag = True
            return 0  # End episode with 0 reward on collision

        # Make Stop Flag
        # stop_flag = (
        #     1
        #     if traffic_light
        #     in [CarlaTrafficLightState.RED, CarlaTrafficLightState.YELLOW]
        #     else 0
        # )

        # Filter objects
        # filtered_objects = [
        #     obj
        #     for obj in object_list
        #     if obj.type not in [ObjectType.TRAFFIC_LIGHT, ObjectType.TRAFFIC_SIGN]
        # ]

        # Speed Reward Calculation
        safe_distance = 0
        speed_reward = 0
        max_speed = (
            speed_limit + speed_margin
        )  # Example max speed limit for reward scaling
        safe_distance_reward = 0
        if not vehicle_in_front:  # No objects in front
            if current_speed == 0:
                speed_reward = 0  # No reward for being stationary
            elif 0 < current_speed <= speed_limit:
                speed_reward = min(
                    1, current_speed / speed_limit
                )  # Linearly scale up to the speed limit
            elif speed_limit < current_speed < max_speed:
                speed_reward = max(
                    0, (max_speed - current_speed) / (max_speed - speed_limit)
                )  # Linearly ramp down
            elif current_speed >= max_speed:
                speed_reward = 0  # No reward if speed exceeds or equals max_speed

        # Safe Distance Reward Calculation
        # for obj in filtered_objects:
        #     if (
        #         abs(obj.angle) <= np.radians(2) and obj.distance != 0
        #     ):  # Object directly in front
        # elif vehicle_in_front.distance <= max_safe_distance and current_speed < speed_limit + speed_margin:

        elif (
            distance_to_car_in_front <= max_safe_distance
            and current_speed < speed_limit + speed_margin
        ):
            if current_speed > 5.4:
                safe_distance = 2 * (
                    current_speed / 3.6
                )  # Safe distance = 2 seconds of travel in m/s
            else:
                safe_distance = 3

            lower_bound = safe_distance - safe_distance * (safe_distance_margin - 0.10)
            upper_bound = safe_distance + safe_distance * (safe_distance_margin - 0.10)

            if lower_bound <= distance_to_car_in_front <= upper_bound:
                safe_distance_reward = 1  # Perfect safe distance
            elif distance_to_car_in_front < lower_bound:
                # safe_distance_reward = max(0, (distance_to_car_in_front - 3) / lower_bound)  # Linearly decrease to 0
                safe_distance_reward = max(
                    0, (distance_to_car_in_front) / lower_bound
                )  # Linearly decrease to 0
            elif distance_to_car_in_front > upper_bound:
                safe_distance_reward = max(
                    0,
                    1
                    - (distance_to_car_in_front - upper_bound)
                    / (max_safe_distance - upper_bound),
                )

        reward = speed_reward if not vehicle_in_front else safe_distance_reward
        # reward = speed_reward
        # Ensure reward is in range [0, 1]
        reward = np.clip(reward, 0, 1)

        # Smoother Driving Reward Calculation
        acceleration = current_speed - self.last_speed  # / self.dt
        smoothness_penalty = min(
            1.0, np.abs(acceleration) / 10.0
        )  # Normalized penalty for high acceleration/deceleration
        smooth_driving_reward = 1.0 - smoothness_penalty

        reward = reward * smooth_driving_reward

        self.last_speed = (
            current_speed  # reset value of last speed, used to calculate acceleration!
        )

        information = {
            "speed_reward": speed_reward,
            "safe_distance_reward": safe_distance_reward,
            "reward": reward,
            "speed_limit": speed_limit,
            "detected_speed_limit": self.detected_speed_limit,
            "current_speed": current_speed,
            "collisions": self.collisionCounter,
            "distance_to_car_infront": distance_to_car_in_front,
            "detected_distance": self.detected_distance,
            "detected_vehicle": self.detected_vehicle,
            "smoothness_scale": smooth_driving_reward,
            "should_stop": self.should_stop,
            "distance_to_stop": self.distance_to_stop,
            "action_taken": self.action_taken,
            # if vehicle_in_front is not None
            # else "None",
        }
        # Logging for debugging and analysis
        wandb.log(information)
        self.visuals.information = information

        objects = [
            ObjectDTO.from_object(obj, is_relevant=detected_car == obj)
            for obj in object_list_imperfect
        ]
        self.visuals.detected_objects = objects

        return reward