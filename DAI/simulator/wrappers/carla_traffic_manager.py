import carla
from loguru import logger


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
