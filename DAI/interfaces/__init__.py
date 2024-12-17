"""Module that defines abstract types to be used throughout the whole module"""

from .data_carriers import (  # noqa: F401
    AgentFeatures,
    BoundingBox,
    CarlaData,
    CarlaObservation,
    Object,
    ObjectType,
)
from .image import Depth, Image  # noqa: F401
from .system import (  # noqa: F401
    ComputerVisionModule,
    CruiseControlAgent,
    FeatureExtractor,
)
from .world import World  # noqa: F401
