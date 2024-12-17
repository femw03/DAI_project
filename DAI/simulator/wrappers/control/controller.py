# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""This module contains PID controllers to perform lateral and longitudinal control."""

import math
from collections import deque

import numpy as np

from ..carla_core import CarlaVehicle, CarlaVehicleControl
from ..carla_utils import CarlaLocation, CarlaVector3D, CarlaWaypoint


class VehiclePIDController:
    """
    VehiclePIDController is the combination of two PID controllers
    (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """

    def __init__(
        self,
        vehicle: CarlaVehicle,
        args_lateral,
        max_steering=0.8,
    ):
        """
        Constructor method.

        :param vehicle: actor to apply to local planner logic onto
        :param args_lateral: dictionary of arguments to set the lateral PID controller
        using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        :param offset: If different than zero, the vehicle will drive displaced from the center line.
        Positive values imply a right offset while negative ones mean a left one. Numbers high enough
        to cause the vehicle to drive through other lanes might break the controller.
        """

        self.max_steer = max_steering

        self._vehicle = vehicle
        self.past_steering = self._vehicle.control.steer
        self._lat_controller = PIDLateralController(self._vehicle, **args_lateral)

    def run_step(self, waypoint: CarlaWaypoint) -> CarlaVehicleControl:
        """
        Execute one step of control invoking both lateral and longitudinal
        PID controllers to reach a target waypoint
        at a given target_speed.

            :param waypoint: target location encoded as a waypoint
            :return: controls
        """

        current_steering = self._lat_controller.run_step(waypoint)
        control = CarlaVehicleControl.new()
        # Steering regulation: changes cannot happen abruptly, can't steer too much.

        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1

        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering

        return control


class PIDLateralController:
    """
    PIDLateralController implements lateral control using a PID.
    """

    def __init__(self, vehicle: CarlaVehicle, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param offset: distance to the center line. If might cause issues if the value
                is large enough to make the vehicle invade other lanes.
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._e_buffer = deque(maxlen=10)

    def run_step(self, waypoint: CarlaWaypoint) -> float:
        """
        Execute one step of lateral control to steer
        the vehicle towards a certain waypoin.

            :param waypoint: target waypoint
            :return: steering control in the range [-1, 1] where:
            -1 maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control(waypoint, self._vehicle.transform)

    def _pid_control(self, waypoint: CarlaVehicle, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the PID equations

            :param waypoint: target waypoint
            :param vehicle_transform: current transform of the vehicle
            :return: steering control in the range [-1, 1]
        """
        # Get the ego's location and forward vector
        ego_loc = CarlaLocation.from_native(vehicle_transform.location)
        v_vec = CarlaVector3D(vehicle_transform.get_forward_vector())
        v_vec = np.array([v_vec.x, v_vec.y, 0.0])

        w_loc = CarlaLocation.from_native(waypoint.transform.location)

        w_vec = np.array([w_loc.x - ego_loc.x, w_loc.y - ego_loc.y, 0.0])

        wv_linalg = np.linalg.norm(w_vec) * np.linalg.norm(v_vec)
        if wv_linalg == 0:
            _dot = 1
        else:
            _dot = math.acos(np.clip(np.dot(w_vec, v_vec) / (wv_linalg), -1.0, 1.0))
        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip(
            (self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0
        )
