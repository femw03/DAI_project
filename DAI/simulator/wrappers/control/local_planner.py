# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module contains a local planner to perform low-level waypoint following based on PID controllers.
Its main utility lies in the run_step methods which provides the necessary steering controlers
To follow the global route. To set the global route use the global_planner trace_route function
"""

import random
from collections import deque
from enum import IntEnum
from typing import Deque, List, Tuple

from ..carla_core import CarlaVehicle, CarlaVehicleControl
from ..carla_utils import CarlaWaypoint
from .controller import VehiclePIDController


class RoadOption(IntEnum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.

    """

    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class LocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a
    trajectory of waypoints that is generated on-the-fly.

    The low-level motion of the vehicle is computed by using two PID controllers,
    one is used for the lateral control and the other for the longitudinal control (cruise speed).

    When multiple paths are available (intersections) this local planner makes a random choice,
    unless a given global plan has already been specified.
    """

    def __init__(self, vehicle: CarlaVehicle, dt: float):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param opt_dict: dictionary of arguments with different parameters:
            dt: time between simulation steps
            target_speed: desired cruise speed in Km/h
            sampling_radius: distance between the waypoints part of the plan
            lateral_control_dict: values of the lateral PID controller
            longitudinal_control_dict: values of the longitudinal PID controller
            max_throttle: maximum throttle applied to the vehicle
            max_brake: maximum brake applied to the vehicle
            max_steering: maximum steering applied to the vehicle
            offset: distance between the route waypoints and the center of the lane
        :param map_inst: carla.Map instance to avoid the expensive call of getting it.
        """
        self._vehicle = vehicle
        self._world = self._vehicle.world
        self._map = self._world.map

        self._vehicle_controller = None
        self.target_waypoint = None
        self.target_road_option = None

        self._waypoints_queue: Deque[Tuple[CarlaWaypoint, RoadOption]] = deque(
            maxlen=10000
        )
        self._min_waypoint_queue_length = 100
        self._stop_waypoint_creation = False

        # Base parameters
        self._dt = dt
        self._sampling_radius = 2.0
        self._args_lateral_dict = {"K_P": 1.95, "K_I": 0.05, "K_D": 0.2, "dt": self._dt}
        self._max_steer = 0.8
        self._base_min_distance = 3.0
        self._distance_ratio = 0.5

        # initializing controller
        self._vehicle_controller = VehiclePIDController(
            self._vehicle,
            args_lateral=self._args_lateral_dict,
            max_steering=self._max_steer,
        )

        # Compute the current vehicle waypoint
        current_waypoint = self._map.get_waypoint(self._vehicle.location)
        self.target_waypoint, self.target_road_option = (
            current_waypoint,
            RoadOption.LANEFOLLOW,
        )
        self._waypoints_queue.append((self.target_waypoint, self.target_road_option))

    def _compute_next_waypoints(self, k=1) -> None:
        """
        Add new waypoints to the trajectory queue.

        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.get_next(self._sampling_radius))

            if len(next_waypoints) == 0:
                break
            elif len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                road_options_list = _retrieve_options(next_waypoints, last_waypoint)
                index = random.choice(range(len(road_options_list)))
                road_option = road_options_list[index]
                next_waypoint = next_waypoints[index]

            self._waypoints_queue.append((next_waypoint, road_option))

    def set_global_plan(
        self,
        current_plan: List[Tuple[CarlaWaypoint, RoadOption]],
        stop_waypoint_creation=True,
        clean_queue=True,
    ):
        """
        Adds a new plan to the local planner. A plan must be a list of [carla.Waypoint, RoadOption] pairs
        The 'clean_queue` parameter erases the previous plan if True, otherwise, it adds it to the old one
        The 'stop_waypoint_creation' flag stops the automatic creation of random waypoints

        :param current_plan: list of (carla.Waypoint, RoadOption)
        :param stop_waypoint_creation: bool
        :param clean_queue: bool
        :return:
        """
        if clean_queue:
            self._waypoints_queue.clear()

        # Remake the waypoints queue if the new plan has a higher length than the queue
        new_plan_length = len(current_plan) + len(self._waypoints_queue)
        if new_plan_length > self._waypoints_queue.maxlen:
            new_waypoint_queue = deque(maxlen=new_plan_length)
            for wp in self._waypoints_queue:
                new_waypoint_queue.append(wp)
            self._waypoints_queue = new_waypoint_queue

        for elem in current_plan:
            self._waypoints_queue.append(elem)

        self._stop_waypoint_creation = stop_waypoint_creation

    def run_step(self):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return: control to be applied
        """

        # Add more waypoints too few in the horizon
        if (
            not self._stop_waypoint_creation
            and len(self._waypoints_queue) < self._min_waypoint_queue_length
        ):
            self._compute_next_waypoints(k=self._min_waypoint_queue_length)

        # Purge the queue of obsolete waypoints
        veh_location = self._vehicle.location
        vehicle_speed = self._vehicle.velocity.magnitude / 3.6  # m/s
        self._min_distance = (
            self._base_min_distance + self._distance_ratio * vehicle_speed
        )

        num_waypoint_removed = 0
        for waypoint, _ in self._waypoints_queue:
            if len(self._waypoints_queue) - num_waypoint_removed == 1:
                min_distance = 1  # Don't remove the last waypoint until very close by
            else:
                min_distance = self._min_distance

            if veh_location.distance_to(waypoint.transform.location) < min_distance:
                num_waypoint_removed += 1
            else:
                break

        if num_waypoint_removed > 0:
            for _ in range(num_waypoint_removed):
                self._waypoints_queue.popleft()

        # Get the target waypoint and move using the PID controllers. Stop if no target waypoint
        if len(self._waypoints_queue) == 0:
            control = CarlaVehicleControl.new()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False
        else:
            self.target_waypoint, self.target_road_option = self._waypoints_queue[0]
            control = self._vehicle_controller.run_step(self.target_waypoint)

        return control

    def get_plan(self):
        """Returns the current plan of the local planner"""
        return self._waypoints_queue

    def done(self):
        """
        Returns whether or not the planner has finished

        :return: boolean
        """
        return len(self._waypoints_queue) == 0


def _retrieve_options(
    list_waypoints: List[CarlaWaypoint], current_waypoint: CarlaWaypoint
) -> List[RoadOption]:
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.get_next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def _compute_connection(
    current_waypoint: CarlaWaypoint, next_waypoint: CarlaWaypoint, threshold=35
) -> RoadOption:
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n: float = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c: float = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < threshold or diff_angle > (180 - threshold):
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT
