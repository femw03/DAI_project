import carla
from loguru import logger

from .carla_core import CarlaVehicle


class CarlaTrafficManager:
    def __init__(self, traffic_manager: carla.TrafficManager) -> None:
        assert isinstance(traffic_manager, carla.TrafficManager)
        self.traffic_manager = traffic_manager
        self._synchronous_mode = False

    @property
    def synchronous_mode(self) -> bool:
        return self._synchronous_mode

    @synchronous_mode.setter
    def synchronous_mode(self, value: bool) -> None:
        logger.info(f"Setting the traffic manager synchronous mode {value}")
        self.traffic_manager.set_synchronous_mode(value)
        self._synchronous_mode = value

    def ignore_traffic_lights(self, vehicle: CarlaVehicle, should_ignore: bool) -> None:
        self.traffic_manager.ignore_lights_percentage(
            vehicle.actor, 100 if should_ignore else 0
        )

    def ignore_traffic_signs(self, vehicle: CarlaVehicle, should_ignore: bool) -> None:
        self.traffic_manager.ignore_signs_percentage(
            vehicle.actor, 100 if should_ignore else 0
        )

    def ignore_vehicles(self, vehicle: CarlaVehicle, should_ignore: bool) -> None:
        self.traffic_manager.ignore_vehicles_percentage(
            vehicle.actor, 100 if should_ignore else 0
        )

    def ignore_walkers(self, vehicle: CarlaVehicle, should_ignore: bool) -> None:
        self.traffic_manager.ignore_walkers_percentage(
            vehicle.actor, 100 if should_ignore else 0
        )

    def ignore_speed_limit(self, vehicle: CarlaVehicle, should_ignore: bool) -> None:
        self.traffic_manager.vehicle_percentage_speed_difference(
            vehicle.actor, -1 if should_ignore else 30
        )
