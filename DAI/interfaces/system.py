"""Contains abstract definitions of the system components that need to be implemented"""

from abc import ABC, abstractmethod

from .data_carriers import AgentFeatures, CarlaData, CarlaObservation


class ComputerVisionModule(ABC):
    """
    This is the service that contains a method to convert CarlaData into CarlaFeatures.
    """

    @abstractmethod
    def process_data(self, data: CarlaData) -> CarlaObservation:
        """From the data from the Carla Simulator return a set of features"""
        pass


class CruiseControlAgent(ABC):
    """A CruiseControlAgent converts CarlaFeatures into a speed"""

    @abstractmethod
    def get_action(self, state: AgentFeatures) -> float:
        """Calculate the throttle speed (0 - 1) from the given state"""


class FeatureExtractor(ABC):
    """A class that takes features and converts them into the output they require, they may contain an internal state"""

    @abstractmethod
    def extract(self, observation: CarlaObservation) -> AgentFeatures:
        """extracts the features from the observation"""
        pass

    @abstractmethod
    def reset(self):
        """reset the internal state"""
        pass
