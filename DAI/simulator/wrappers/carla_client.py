from typing import List, Literal, Optional, Tuple
import carla
import os
import logging

from .carla_world import CarlaWorld
from .carla_command import CarlaCommand, CarlaCommandResponse

logger = logging.getLogger(__name__)

IS_DOCKER = os.environ["IS_CONTAINER"] == "TRUE"


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


def main():
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    client = CarlaClient()
    print(client.version)


if __name__ == "__main__":
    main()
