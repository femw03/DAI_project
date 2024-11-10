from DAI.simulator.wrappers.carla_client import CarlaClient

import logging

logging.basicConfig(level=logging.INFO)

print(CarlaClient().version)
