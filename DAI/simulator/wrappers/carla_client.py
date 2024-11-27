import os
from typing import List, Literal, Optional, Tuple

import carla
from loguru import logger

from .carla_core import CarlaWorld
from .carla_traffic_manager import CarlaTrafficManager
from .carla_utils import CarlaCommand, CarlaCommandResponse

IS_DOCKER = "IS_CONTAINER" in os.environ and os.environ["IS_CONTAINER"] == "TRUE"


class CarlaClient:
    """Proxy class that adds typing support for carla"""

    def __init__(
        self,
        host: Literal["localhost", "host.docker.internal"] = None,
        port=2000,
        timeout=20,
    ) -> None:
        if host is None:
            host = "host.docker.internal" if IS_DOCKER else "localhost"
        try:
            logger.info(f"Connecting to carla client at {host}:{port}")
            self.client = carla.Client(host, port)
            self.client.set_timeout(timeout)
            logger.info(
                f"Successfully connected to carla\nclient version: {self.version[0]}\nserver version: {self.version[1]}"
            )
        except Exception as e:
            logger.error("An exception occured while connecting")
            raise e

        self.world: Optional[CarlaWorld] = CarlaWorld(self.client.get_world())

    @property
    def version(self) -> Tuple[str, str]:
        """Get the version from client and server"""
        return self.client.get_client_version(), self.client.get_server_version()

    def load_world(self, map_name: str) -> CarlaWorld:
        self.world = CarlaWorld(self.client.load_world(map_name))

    def apply_batch_sync(
        self, commands: List[CarlaCommand]
    ) -> List[CarlaCommandResponse]:
        responses = self.client.apply_batch_sync(
            [command.reduced for command in commands]
        )
        return [CarlaCommandResponse(response) for response in responses]

    def get_traffic_manager(self) -> CarlaTrafficManager:
        return CarlaTrafficManager(self.client.get_trafficmanager(8000))


def main():
    client = CarlaClient()
    print(client.version)


if __name__ == "__main__":
    main()
