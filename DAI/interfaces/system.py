"""Contains abstract definitions of the system components that need to be implemented"""

from abc import ABC, abstractmethod

from .data_carriers import CarlaData, CarlaFeatures


class ComputerVisionModule(ABC):
    """
    This is the service that contains a method to convert CarlaData into CarlaFeatures.
    """

    @abstractmethod
    def process_data(self, data: CarlaData) -> CarlaFeatures:
        """From the data from the Carla Simulator return a set of features"""
        pass


class CruiseControlAgent(ABC):
    """A CruiseControlAgent converts CarlaFeatures into a speed"""

    @abstractmethod
    def get_action(self, state: CarlaFeatures) -> float:
        """Calculate the throttle speed (0 - 1) from the given state"""
