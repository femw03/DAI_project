import carla
import logging

logger = logging.getLogger(__name__)


class CarlaActor:
    """Wrapper class around a carla actor"""

    def __init__(self, actor: carla.Actor) -> None:
        assert isinstance(
            actor, carla.Actor
        ), f"Instantiate a CarlaActor with a carla.Actor object instead of {type(actor)}"
        self.actor = actor
