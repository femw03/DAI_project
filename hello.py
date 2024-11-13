from DAI.simulator import World

import logging

logging.basicConfig(level=logging.INFO)
world = World()
print(world)
world.start()
